"""extensions of rust_circuit"""

# %%

import rust_circuit as rc

import torch
from typing import Optional, Union, Sequence

from pathlib import Path
import collections
import itertools
import textwrap
import copy

# a reasonable default printer
printer_commenters = [
    lambda c: f'exp_comp={int(c.is_explicitly_computable)}',
    lambda c: f'hash={hex(hash(c.hash))}',
]
printer = rc.PrintHtmlOptions(
    shape_only_when_necessary=False,
    comment_arg_names=True,
    commenters=printer_commenters
)

def clear_cuda_cache():
    """clear various cuda-related caches"""
    rc.clear_module_circuit_caches()
    torch.cuda.empty_cache()

def check_is_leaf_or_var(c: rc.Circuit) -> bool:
    """check if `c` is a leaf or a random var over a leaf"""
    if c.is_leaf():
        return True
    if c.is_discrete_var():
        return c.children[0].is_leaf()
    return False

def assert_can_be_input(c: rc.Circuit):
    """assert that a node can be used as an input"""
    if not check_is_leaf_or_var(c):
        raise RuntimeError(f'expect an input node, got {c}')

def expand_with_batch(
        c: rc.Circuit, inp_names: Union[list[str], str], batch_size: int,
        name_suffix: str = 'batch') -> tuple[
            rc.Circuit,
            Union[list[rc.Symbol], rc.Symbol]]:
    """expand the circuit with batch dim

    :param inp_names: list of input names or the input name if there is only one
        input
    :return: new circuit, input symbol. If ``input_names`` is a str, return a
        single batch symbol; otherwise return a list of batch symbols
    """
    if isinstance(inp_names, str):
        inp_names = [inp_names]
        return_single = True
    else:
        return_single = False
    spec = []
    batch_syms = []
    for i in inp_names:
        assert isinstance(i, str)
        sym_in = c.get_unique(i)
        assert_can_be_input(sym_in)

        new_shape = (batch_size, ) + sym_in.shape
        new_name = name=f'{i}_{name_suffix}'

        # Something very stupid:
        # 1. rc.Symbol does not have dtype or device information
        # 2. We can use a GeneralFunction to implement a type-aware placeholder
        #    op, but rc.Schedule does not replace tensors from GeneralFunction
        #
        # Therefore, we use rc.Symbol if possible, and use an empty array
        # otherwise

        if not sym_in.torch_dtype or sym_in.torch_dtype == torch.float32:
            sym_in_batch = rc.Symbol.new_with_random_uuid(new_shape, new_name)
        else:
            sym_in_batch = make_placeholder(
                new_shape, sym_in.device, sym_in.torch_dtype, new_name)
        batch_syms.append(sym_in_batch)
        spec.append((sym_in, lambda _, *, b=sym_in_batch: b))
    expander = rc.Expander(*spec)
    if return_single:
        batch_syms, = batch_syms
    return rc.conform_all_modules(expander(c)), batch_syms


class PrintOp(rc.GeneralFunctionSpecBase):
    """an op that prints the tensor value for debug"""

    def __init__(self, name):
        self._name = name
        self._hash = abs(hash(name)).to_bytes(8, 'little')

    @property
    def path(self) -> str:
        return f'{Path(__file__).resolve()}:{self.__class__.__name__}'

    @property
    def name(self) -> str:
        return self._name

    def compute_hash_bytes(self) -> bytes:
        return self._hash

    def function(self, x: torch.Tensor) -> torch.Tensor:
        print(f'========== print {self._name} ')
        print(x)
        return x

    def get_shape_info(self, *shapes: rc.Shape) -> rc.GeneralFunctionShapeInfo:
        return rc.get_shape_info_simple(shapes)

    @classmethod
    def new(cls, circ: rc.Circuit, name):
        return rc.GeneralFunction(circ, spec=cls(name), name=name)


def replace_nodes(c: rc.Circuit,
                  mapping: dict[rc.Circuit, rc.Circuit]) -> rc.Circuit:
    """A safe node replacement procedure: replace exactly the nodes given in the
    dict (matched by node hash, not by name); also check that all nodes are
    actually replaced
    """
    # also see this discussion on slack:
    # https://remix-lpn3709.slack.com/archives/C04HLB25L92/p1674163257656589

    # we can return the original c, but add this assert just to be safe
    assert len(mapping), 'mapping should not be empty'

    used = collections.defaultdict(int)
    def visit(x: rc.Circuit):
        y = mapping.get(x, None)
        if y is None:
            return x
        used[x] += 1
        return y
    cnew = rc.deep_map_preorder(c, visit)

    if len(used) != len(mapping):
        for i in mapping.keys():
            if i not in used:
                raise RuntimeError(f'key {i} in mapping not replaced')

    assert (max(used.values()) == 1 and len(used) == len(mapping) and
            set(used.keys()) == set(mapping.keys()))
    return cnew

def fmt_node_list(nodes: Sequence[rc.Circuit]) -> str:
    """format a list of nodes to a non-redwood-human-readable string"""
    return (f'{len(nodes)}' +
            '\n '.join(sorted(itertools.chain(
                [''], (f'{i.name} {i.shape}' for i in nodes)))))

def make_placeholder(shape, device, dtype, name) -> rc.Circuit:
    """make an explicitly computable placeholder node with device and dtype
    information"""
    return rc.Array(
        torch.tensor(0, dtype=dtype, device=device).broadcast_to(shape),
        name=name
    )

class PartialCircuit:
    """evaluate a part of a circuit by binding some inputs"""

    _boundary_syms: list[rc.Leaf]
    """symbols for the boundary nodes"""

    _partial_schedule: rc.Schedule
    """schedule to evaluate values of boundary nodes"""

    _apply_schedule: rc.Schedule
    """schedule to evaluate the output given boundary nodes and other inputs"""

    _apply_extra_partial_inp_hash: set[bytes]
    """partial inputs needed in the apply circuit"""

    _partial_inp_nodes: list[rc.Circuit]
    _other_inp_syms: list[rc.Leaf]

    def __init__(self, out: rc.Circuit,
                 partial_inp: list[rc.Leaf], other_inp: list[rc.Symbol],
                 optim_settings=rc.OptimizationSettings()):
        assert (
            len(set(i.name for i in itertools.chain(partial_inp, other_inp))) ==
            len(partial_inp) + len(other_inp)), (
                'inputs must have distince names')
        for i in partial_inp:
            assert_can_be_input(i)

        for i in other_inp:
            assert i.is_symbol(), (
                'other inp must be Symbol so they are not explictly computable,'
                f' got {i}')

        def get_device(d):
            # we replace the inputs with Array to infer which part of the graph
            # can be calculated during binding. Array requires a device. However,
            # ``rc.Symbol`` (perhaps with some other nodes too) do not have an
            # associated device. In this case, we will use the device of the
            # output as the device of those newly created arrays as a best
            # guess. This may cause problems if the circuit is split on multiple
            # devices (not sure if rc supports this though).
            if d is None:
                return out.device
            return d

        def get_dtype(d):
            if d is None:
                return torch.float32
            return d

        def placeholder_like(x):
            return make_placeholder(
                x.shape, get_device(x.device), get_dtype(x.torch_dtype),
                name=f'{x.name}_{name_suffix}')

        # a unique id to avoid name conflict
        name_suffix = f'placeholder_{hex(id(self))}'

        # replace the partial inputs to reuse rc's is_explicitly_computable
        partial_repl = {i: placeholder_like(i) for i in partial_inp}
        out_repl = replace_nodes(out, partial_repl)
        assert not out_repl.is_explicitly_computable, (
            'output does not depend on other inputs')

        boundary_nodes = set()  # use a set to deduplicate
        visited = set()

        partial_inp_set_repl = set(partial_repl.values())
        apply_needed_inp = set()

        def find_boundary(root: rc.Circuit):
            if root in visited:
                return
            assert not (root.is_discrete_var() or root.is_cumulant()
                        or root.is_stored_cumulant_var()), (
                f'partial circuit does not support random vars: {root}'
            )
            visited.add(root)
            ch = root.children
            if root.is_module():
                ch = ch[1:]

            for i in ch:
                if i.is_explicitly_computable:
                    if i.is_leaf():
                        if i in partial_inp_set_repl:
                            apply_needed_inp.add(i)
                    else:
                        boundary_nodes.add(i)
                else:
                    find_boundary(i)

        find_boundary(out_repl)
        print(f'{self.__class__.__name__}: '
              f'partial inp for apply: {len(apply_needed_inp)};'
              f' boundary nodes: {fmt_node_list(boundary_nodes)}')

        def print_circuit(circ):
            circ.print(printer.evolve(commenters=printer_commenters + [
                lambda c: f'vis={int(c in visited)}',
                lambda c: f'leaf={int(c.is_leaf())}',
                lambda c: f'bnd={int(c in bnd)}',
            ]))

        # call print here for debug

        if not boundary_nodes:
            print_circuit(out_repl)
            raise RuntimeError(
                'partial function does not depend on other_inp;'
                f' visited={len(visited)}'
            )
        boundary_nodes = list(boundary_nodes)   # list for stable order
        self._partial_schedule = rc.optimize_to_schedule_many(
            boundary_nodes, settings=optim_settings)

        boundary_syms = {i: placeholder_like(i) for i in boundary_nodes}
        out_apply = replace_nodes(out_repl, boundary_syms)
        self._apply_schedule = rc.optimize_to_schedule(
            out_apply, settings=optim_settings)
        self._apply_extra_partial_inp_hash = set(
            i.hash for i in apply_needed_inp)

        self._partial_inp_nodes = list(map(partial_repl.__getitem__,
                                           partial_inp))
        self._other_inp_syms = other_inp
        self._boundary_syms = list(map(boundary_syms.__getitem__,
                                       boundary_nodes))

    class BoundCircuit:
        """a ciruit with values of partial inputs given"""
        _schedule: rc.Schedule
        _inps: list[rc.Leaf]
        _extra_kv: dict[bytes, torch.Tensor]

        def __init__(self, schedule, inps, extra_kv):
            self._schedule = schedule
            self._inps = inps
            self._extra_kv = extra_kv

        def __call__(self, inp_vals: list[torch.Tensor]) -> torch.Tensor:
            """evaluate the original output given the values of other inputs"""
            assert len(inp_vals) == len(self._inps)
            for i, j in zip(self._inps, inp_vals):
                assert i.shape == j.shape, (
                    f'shape mismatch: {i=} {i.shape=} {j.shape=}')
            kv = {i.hash: j for i, j in zip(self._inps, inp_vals)}
            kv.update(self._extra_kv)
            s = self._schedule.replace_tensors(kv)
            return s.evaluate()


    def bind(self, partial_inp_vals: list[torch.Tensor]) -> BoundCircuit:
        assert len(partial_inp_vals) == len(self._partial_inp_nodes)
        for i, j in zip(self._partial_inp_nodes, partial_inp_vals):
            assert i.shape == j.shape, (
                f'shape mismatch: {i=} {i.shape=} {j.shape=}')
        repl = {i.hash: j
                for i, j in zip(self._partial_inp_nodes, partial_inp_vals)}

        s = self._partial_schedule.replace_tensors(repl)
        boundary_vals = s.evaluate_many()
        assert len(boundary_vals) == len(self._boundary_syms)

        repl = {i: j for i, j in repl.items()
                if i in self._apply_extra_partial_inp_hash}
        repl.update((i.hash, j)
                    for i, j in zip(self._boundary_syms, boundary_vals))
        return self.BoundCircuit(
            self._apply_schedule, self._other_inp_syms, repl)


class BeautifulDiff:
    """a diff tool that goes to the different leaf directly"""
    _diff_roots: dict[tuple[rc.Circuit, rc.Circuit], list[list[str]]]
    """roots of the subtrees that are different"""

    def __init__(self) -> None:
        self._diff_roots = collections.defaultdict(list)

    def compute(self, tree1: rc.Circuit, tree2: rc.Circuit):
        if tree1 == tree2:
            return self
        self._do_compute(tree1, tree2, [tree1.name])
        return self

    def _do_compute(self, tree1: rc.Circuit, tree2: rc.Circuit,
                    path: list[str]):
        """compute, knowing that roots are different"""

        if tree1.name != tree2.name or len(tree1.children) != len(tree2.children):
            self._diff_roots[(tree1, tree2)].append(path)
            return

        diff_children = []
        for i, j in zip(tree1.children, tree2.children):
            if i != j:
                diff_children.append((i, j))

        if len(diff_children) == len(tree1.children) and len(diff_children) > 1:
            # all children are different; show the diff at root
            self._diff_roots[(tree1, tree2)].append(path)
        else:
            path.append(tree1.name)
            for i in diff_children:
                self._do_compute(*i, path)

    def print(self, *args, **kwargs):
        """"``args`` and ``kwargs`` are extra args to be passed to the html
        printer"""
        print(f'Total {len(self._diff_roots)} different roots')
        for idx, ((r1, r2), paths) in enumerate(self._diff_roots.items()):
            print(f'{idx+1}/{len(self._diff_roots)}', '='*70)
            print(f'Paths:\n {self.fmt_paths(paths)}')
            r1.print_html(*args, **kwargs)
            r2.print_html(*args, **kwargs)

    @classmethod
    def fmt_paths(cls, paths: list[list[str]]) -> str:
        lines = []
        paths = copy.deepcopy(paths)
        maxlen = max(map(len, paths))
        for i in paths:
            i.extend([''] * (maxlen - len(i)))
        p0 = paths[0]
        for i in range(len(p0)):
            if any(len(pj) == i or pj[i] != p0[i] for pj in paths):
                lines.append(', '.join(p0[:i]))
                for j, pj in enumerate(paths):
                    lines.append(f'  {j}:' + ', '.join(pj[i:]))
        if not lines:
            lines = [', '.join(p0)]

        for i, j in enumerate(lines):
            lines[i] = '\n'.join(textwrap.wrap(j, width=70, subsequent_indent='   '))
        return '\n'.join(lines)


def beautiful_diff(c1: rc.Circuit, c2: rc.Circuit, *args, **kwargs):
    """"``args`` and ``kwargs`` are extra args to be passed to the html
    printer"""
    BeautifulDiff().compute(c1, c2).print(*args, **kwargs)


class ScrubPrinter:
    """colorize nodes dpending on if their usage of two provided vars"""
    baseline_name: str
    other_name: str

    baseline_var: rc.Circuit
    other_var: rc.Circuit

    _hash2type: dict[bytes, str]

    def scrubbed(self, c: rc.Circuit):
        return c.are_any_found(self.other_var)

    def not_scrubbed(self, c: rc.Circuit):
        return c.are_any_found(self.baseline_var)

    def get_scrub_type(self, c: rc.Circuit) -> str:
        t = self._hash2type.get(c.hash)
        if t is not None:
            return t
        getting_scrubbed = self.scrubbed(c)
        getting_unscrubbed = self.not_scrubbed(c)
        if getting_scrubbed and getting_unscrubbed:
            t = 'both'
        elif getting_scrubbed:
            t = 'other_inp'
        elif getting_unscrubbed:
            t = 'target_inp'
        else:
            t = 'unrelated'
        self._hash2type[c.hash] = t
        return t

    def __init__(self, baseline_name: str, other_name: str):
        self.baseline_name = baseline_name
        self.other_name = other_name

    def __call__(self, c: rc.Circuit) -> rc.Circuit:
        """print the circuit and return c"""
        self.baseline_var = c.get_unique(self.baseline_name)
        self.other_var = c.get_unique(self.other_name)
        self._hash2type = {}

        color_map = {
            'both': 'purple',
            'other_inp': 'red',
            'target_inp': 'cyan',
            'unrelated': 'lightgrey',
        }

        scrubbed_printer = printer.evolve(
            colorer=lambda c: color_map[self.get_scrub_type(c)],
            traversal=rc.restrict(
                printer.traversal,
                term_early_at=lambda c: self.get_scrub_type(c) in [
                    'target_inp', 'unrelated']
            ),
            commenters=[self.get_scrub_type],
        )
        c.print(scrubbed_printer)
        return c


def run_tests():
    xv = rc.Array(torch.tensor([0]), name='xv')

    xv1 = rc.Array(torch.tensor([1]), name='xv1')

    try:
        replace_nodes(xv1, {xv: xv1})
        assert 0, 'exception not raised'
    except RuntimeError:
        pass

    assert replace_nodes(xv1.add(xv, name='+'), {xv: xv1}) == xv1.add(
        xv1, name='+')

    # at test: x=1, y=2
    yv = rc.Symbol.new_with_random_uuid((1, ), name='yv')
    x = PrintOp.new(xv, name='prx') # 1
    x1 = PrintOp.new(x.add(rc.Scalar(2), name='x+2'), name='pr x+2')    # 3
    z = (PrintOp.new(x.mul_scalar(2, name='x*2').add(yv, name='x*2+y'),
                     name='pr x*2+y')   # 4
         .mul(x1, name='x2y2'))     # 12

    z.print(printer)
    assert x.hash == PrintOp.new(xv, name="prx").hash
    assert x.hash != PrintOp.new(yv, name="prx").hash

    circ = PartialCircuit(z, [xv], [yv])
    print('!!!! before bind')
    bind = circ.bind([torch.tensor([1])])
    print('!!!! after bind')
    result = bind([torch.tensor([2])])
    print('!!!! after eval', result)
    torch.testing.assert_close(result, torch.tensor([12]))

    print('=========== test0')

    z1 = z.add(xv)

    circ = PartialCircuit(z1, [xv], [yv])
    print('!!!! before bind')
    bind = circ.bind([torch.tensor([1])])
    print('!!!! after bind')
    result = bind([torch.tensor([2])])
    print('!!!! after eval', result)
    torch.testing.assert_close(result, torch.tensor([13]))

if __name__ == '__main__':
    run_tests()
