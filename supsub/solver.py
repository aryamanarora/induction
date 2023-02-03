"""solvers integrated with rust_circuit"""

from .projection import ProjectionLayer, AdversarialProjectionMatrix
from .rc_ext import (
    replace_nodes, PartialCircuit, assert_can_be_input, fmt_node_list,
    clear_cuda_cache)

from interp.tools.indexer import TORCH_INDEXER as I
import rust_circuit as rc
import torch
import torch.nn.functional as F
import attrs

import typing
import itertools

class FeatureReconProjectionSolver:
    """solve the projection by reconstruting feature maps with an L2 loss; fast,
    but not recommended. Use :class:`LookAheadProjectionSolver` if possible.
    This is the zbp solver in the MNIST experiments."""
    proj: ProjectionLayer
    loss_mean: typing.Optional[float]

    _opt: torch.optim.Optimizer
    _device: str

    def __init__(self, dim: int, proj_dim: int, device: str):
        self._device = device
        p = self.proj = ProjectionLayer(dim, proj_dim).to(self._device)
        self.loss_mean = None
        self._opt = torch.optim.AdamW(
            [
                {
                    'params': p.pmat.parameters(),
                    'weight_decay': 1e-4
                },
                {
                    'params': [p.pcenter],
                    'lr': 1.5e-3,
                }
            ]
        )

    def proj_decompose(self, x: torch.Tensor, *, ctx=None) -> tuple[
            torch.Tensor, torch.Tensor]:
        """decompose x into the subspace part and the orthogonal part

        :param pmat: the projection matrix; if not provided, the internal
            projection matrix will be used
        """
        xin = self.proj(x, ctx=ctx)
        xout = x - xin
        return xin, xout

    def update(self, ftr_target: torch.Tensor, ftr_other: torch.Tensor):
        """update the projection with one data iteration"""
        assert ftr_target.shape == ftr_other.shape
        assert ftr_target.ndim == 2 and ftr_target.shape[1] == self.proj.dim, (
            ftr_target.shape, self.proj.dim)

        if self.loss_mean is None:
            self.proj.reset_as_pca(ftr_target)

        ctx = self.proj.ForwardCtx()
        self._opt.zero_grad()
        target_in, _ = self.proj_decompose(ftr_target, ctx=ctx)
        _, other_out = self.proj_decompose(ftr_other, ctx=ctx)
        recon = target_in + other_out
        loss = ((recon - ftr_target)**2).mean()

        if self.loss_mean is None:
            self.loss_mean = loss.item()
        else:
            self.loss_mean = self.loss_mean * .9 + loss.item() * .1

        loss.backward()
        self._opt.step()


@attrs.define(frozen=True)
class CircuitWithProjection:
    """a circuit with certain heads replaced by a projection"""

    circuit: rc.Circuit
    """the circuit with projection operations"""

    proj: ProjectionLayer
    """the projection layer used in the circuit"""

    proj_inp_name: str
    """name of the node that contains combined features before projection"""


@attrs.define(frozen=True)
class AdvSolvingMode:
    """modes for solving the adversary in :class:`LookAheadProjectionSolver`"""

    incr_loss: typing.Optional[bool] = None
    """
    If true, the adversary tries to increase the loss (e.g., adv can
    remove a few dims to make the loss worse); goal of original projection is to
    decrease the loss.

    If false, the adversary tries to increase the loss (e.g., adv can add a few
    dims to improve the projections); goal of the original projection is to
    miminize the improvement.

    If None, determined by whether the adv tries to remove dim
    """

    adv_only: bool = False
    """whether to only solve the adversary; if set to true, will not update the
    original projection, which is useful for evaluating adversarial robustness.
    """

    replace_by_adv: typing.Optional[bool] = None
    """whether :meth:`LookAheadProjectionSolver.replace_with_proj` should use
    the adversary-perturbed projection or the original projection

    if None, determined by :attr:`adv_only`
    """

    def canonize(self, pmat: AdversarialProjectionMatrix) -> "AdvSolvingMode":
        """check the attributes and initialize default values"""
        if self.incr_loss is None:
            is_rm = pmat.num_rm != 0
            is_add = pmat.num_add != 0
            assert is_rm + is_add == 1, (
                'can not infer solving mode; please set '
                'adv_solving_mode.incr_loss')
            self = attrs.evolve(self, incr_loss=is_rm)

        if self.replace_by_adv is None:
            self = attrs.evolve(self, replace_by_adv=self.adv_only)

        assert isinstance(self.incr_loss, bool)
        assert isinstance(self.adv_only, bool)
        assert isinstance(self.replace_by_adv, bool)
        return self


class LookAheadProjectionSolver:
    """solve the projection layer by optimzing the loss through later layers of
    the network. We assume the circuit only takes one input."""

    _logits: rc.Circuit
    """logits with projection nodes replaced"""

    _input_node: rc.Symbol
    """the input node in logits"""

    _sym_proj_srcs: list[rc.Circuit]
    """projection nodes in the original circuit"""

    _sym_proj_repl: list[rc.Symbol]
    """Symbol nodes that have replaced the original nodes in the :attr:`_logits`
    circuit"""

    _matcher: rc.IterativeMatcherIn

    _proj_schedule: rc.Schedule
    _logits_partial: PartialCircuit

    _logits_schedule: typing.Optional[rc.Schedule] = None
    """when the logits computation only depends on the projected value but not
    original input, we have :attr:`_logits_schedule`. Exactly one of
    :attr:`_logits_partial` should be set"""

    _opt: torch.optim.Optimizer
    _opt_adv: torch.optim.Optimizer

    _min_loss_mean: float = float('inf')
    """min :attr:`loss_mean` achieved, for early stop check"""

    _min_loss_mean_iter_num: int = 0
    _min_loss_mean_proj_dim: int = -1

    _iter_num: int = 0
    """number of iterations, i.e., number of calls to :meth:`update`"""

    _adv_solving_mode: AdvSolvingMode

    class SharedComputingContext:
        """see :meth:`LookAheadProjectionSolver.update` for more details"""

        init: bool = False
        """whether it has been initialized"""

        params: tuple
        """input params for safety check"""

        proj_target: torch.Tensor
        """pre-projection computed on the target input"""

        proj_other: torch.Tensor
        """pre-projection computed on the other input"""

        orig_probs: torch.Tensor
        """output probability computed on the target input"""

        logits_bound: typing.Callable[[list[torch.Tensor]], torch.Tensor]
        """a function to compute the logits with non-projection-inputs already
        bound"""


    use_pca_init: bool

    proj: ProjectionLayer
    """the trained projection layer; do not reassign this attribute. Use
    ``load_state_dict()`` if you want to load an existing projection layer"""

    loss_mean: typing.Optional[float] = None
    """moving-average loss"""

    suggest_early_stop: bool = False
    """this flag will be set to True by :meth:`update` if local minimum seems to
    have been reached"""

    def __init__(
            self,
            logits: rc.Circuit,
            matcher: rc.IterativeMatcherIn, proj_dim: int,
            input_name: str,
            optim_settings=rc.OptimizationSettings(),
            proj_args: dict[str, typing.Any] = {},
            adv_solving_mode: typing.Optional[AdvSolvingMode] = None,
            use_pca_init=False):
        """
        :param logits: classification logits. The logits should be at the last
            dimension. It should have already been expanded with batch
            dimension.
        :param matcher: the matcher that starts from logits to select the nodes
            whose output should be replaced by the projection; each node should
            also have (seq, hid_i) shape, with seq corresponds to ``logits``
        :param proj_dim: dimension of the projection space
        :param input_name: name of the input node; we assume a single input
        :param proj_args: keyword arguments passed to :class:`ProjectionLayer`
        :param adv_solving_mode: mode for solving the adversarial projection;
            see :class:`AdvSolvingMode` for more details.
        :param use_pca_init: whether to initialize the projection matrix from
            PCA of the first minibatch
        """
        clear_cuda_cache()
        self._input_node = logits.get_unique(input_name)
        assert_can_be_input(self._input_node)
        self._matcher = matcher

        nodes = self._init_circuit(logits, optim_settings)
        self.proj = ProjectionLayer(
            sum(i.shape[-1] for i in nodes), proj_dim, **proj_args).train()

        print(f'{self.__class__.__name__}: proj={self.proj};'
              f' proj nodes: {fmt_node_list(nodes)}')

        if logits.device:
            self.proj.to(logits.device)

        self._opt = torch.optim.AdamW(itertools.chain(
            self.proj.pmat.parameters(), [self.proj.pcenter]))

        if self.proj.pmat_adv is not None:
            self._opt_adv = torch.optim.AdamW(self.proj.pmat_adv.parameters())

        if adv_solving_mode is None:
            adv_solving_mode = AdvSolvingMode()
        if self.proj.pmat_adv is not None:
            adv_solving_mode = adv_solving_mode.canonize(self.proj.pmat_adv)
        self._adv_solving_mode = adv_solving_mode

        self.use_pca_init = use_pca_init

    def _init_circuit(
            self, logits: rc.Circuit, optim_settings) -> list[rc.Circuit]:
        """initialize the circuits and schedulers
        :return: projection nodes
        """
        matcher = self._matcher
        nodes = list(logits.get(matcher))
        assert nodes, f'empty nodes: {logits=} {matcher=}'
        assert len(set(i.name for i in nodes)) == len(nodes), (
            f'matched nodes have duplicated names: {nodes}'
        )
        nodes.sort(key=lambda n: n.name)    # ensure stable order
        for i in nodes:
            assert i.shape[:-1] == nodes[0].shape[:-1], (
                f'{i}: bad shape: {i.shape} (ref shape {nodes[0].shape})')
        self._sym_proj_srcs = nodes

        nodes_repl = {
            n: rc.Symbol.new_with_none_uuid(n.shape, name=f'proj_{i}_{n.name}')
            for i, n in enumerate(nodes)
        }
        self._logits = logits.update(matcher, nodes_repl.__getitem__)
        self._sym_proj_repl = list(map(nodes_repl.__getitem__, nodes))

        self._proj_schedule = rc.optimize_to_schedule_many(
            nodes, settings=optim_settings)
        if self._logits.get(self._input_node):
            self._logits_partial = PartialCircuit(
                self._logits, [self._input_node], self._sym_proj_repl,
                optim_settings=optim_settings
            )
        else:
            self._logits_schedule = rc.optimize_to_schedule(
                self._logits, settings=optim_settings)

        return nodes

    def _eval_proj_concat(self, inp: torch.Tensor) -> tuple[
            torch.Tensor, list[torch.Tensor]
        ]:
        """eval the values of projection nodes and concat the results

        :return: the nodes concatenated by the last dimension, and the
            individual outputs
        """
        assert inp.shape == self._input_node.shape, (
            f'expected shape {self._input_node.shape}, '
            f'actual shape {inp.shape}; if you neeed to add a batch dim, see '
            f'expand_with_batch'
        )
        vals = self._proj_schedule.replace_tensors(
            {self._input_node.hash: inp}
        ).evaluate_many()
        return torch.cat(vals, dim=-1), vals

    def _split_concated_proj(self, x: torch.Tensor) -> list[torch.Tensor]:
        """split concatenated projection features into individual ones"""
        ret = []
        off = 0
        for i in self._sym_proj_repl:
            s = i.shape[-1]
            ret.append(x[..., off:off+s])
            off += s
        assert off == x.shape[-1]
        return ret

    @torch.no_grad()
    def _init_shared_computing(
            self, ctx: SharedComputingContext,
            inp_target: torch.Tensor, inp_other: torch.Tensor,
            seq_pos_mask: typing.Optional[torch.Tensor]):
        """compute non-projection part and store the result in ``ctx``"""

        params = (inp_target, inp_other, seq_pos_mask, self._logits,
                  self._matcher)
        if ctx.init:
            assert (
                all(i is j for i, j in zip(params[:3], ctx.params[:3]))
                and all(i == j for i, j in zip(params[3:], ctx.params[3:]))
            ), 'can not share compute if input changes'
            return

        ctx.params = params

        proj_target, proj_target_comp = self._eval_proj_concat(inp_target)
        proj_other, _ = self._eval_proj_concat(inp_other)

        if self._logits_schedule is None:
            logits_bound = self._logits_partial.bind([inp_target])
        else:
            def logits_bound(proj_valus: list[torch.Tensor]):
                s = self._logits_schedule.replace_tensors({
                    i.hash: j
                    for i, j in zip(self._sym_proj_repl, proj_valus)
                })
                return s.evaluate()

        orig_logits = logits_bound(proj_target_comp)
        if seq_pos_mask is not None:
            shape0 = orig_logits.shape
            orig_logits = orig_logits[seq_pos_mask]
            assert orig_logits.shape[-1] == shape0[-1]
        orig_probs = F.softmax(orig_logits, dim=-1)

        ctx.proj_target = proj_target
        ctx.proj_other = proj_other
        ctx.orig_probs = orig_probs
        ctx.logits_bound = logits_bound
        ctx.init = True

    def update(self, inp_target: torch.Tensor,
               inp_other: torch.Tensor,
               seq_pos_mask: typing.Optional[torch.Tensor] = None,
               shared_computing: typing.Optional[SharedComputingContext] = None
               ) -> "Self":
        """update the projection layer for one iteration

        :param seq_pos_mask: a boolean mask in the shape (batch, seq) to
            constrain loss computation on certain sequence locations
        :param shared_computing: you may want to run multiple solvers with
            different hyperparameters (projection dim, weight decay, etc.)
            together. In this case, you can use ``shared_computing`` to save
            some time. Just pass the same ``shared_computing`` object to all
            solvers.

            Example::

                solver_1 = LookAheadProjectionSolver(...)
                solver_2 = LookAheadProjectionSolver(...)

                sc = solver_1.SharedComputingContext()
                solver_1.update(x1, x2, shared_computing=sc)
                solver_2.update(x1, x2, shared_computing=sc)
        """
        if shared_computing is None:
            shared_computing = self.SharedComputingContext()

        self._init_shared_computing(shared_computing,
                                    inp_target, inp_other, seq_pos_mask)

        if self._iter_num == 0:
            self.proj.reset_as_pca(shared_computing.proj_target,
                                   bias_only=not self.use_pca_init)

        if self.proj.pmat_adv is not None:
            # update adversary before updating the target
            loss = self._step(shared_computing, seq_pos_mask, True)

        if not self._adv_solving_mode.adv_only:
            loss = self._step(shared_computing, seq_pos_mask, False)

        if self.loss_mean is None:
            self.loss_mean = loss
        else:
            self.loss_mean = self.loss_mean * .9 + loss * .1

        self._iter_num += 1
        if self.loss_mean < self._min_loss_mean:
            self._min_loss_mean = self.loss_mean
            self._min_loss_mean_iter_num = self._iter_num

        # recompute early stop suggestion
        self.suggest_early_stop = False
        if ((self.proj.pmat_adv is None or self._adv_solving_mode.adv_only)
                and self._iter_num >= 20
                and self._iter_num - self._min_loss_mean_iter_num >= 30):
            # adaptive methods may change projection dim, so we check that dim
            # is stable before early stop
            proj_dim = self.proj.proj_dim
            if proj_dim == self._min_loss_mean_proj_dim:
                self.suggest_early_stop = True
            else:
                self._min_loss_mean_proj_dim = proj_dim
                self._min_loss_mean_iter_num = self._iter_num

        return self

    def _step(self, shared_computing: SharedComputingContext,
              seq_pos_mask, is_adv_mode: bool) -> float:
        """run one iteration on the optimizer with given projection function

        :return: loss value
        """
        proj_target = shared_computing.proj_target
        proj_other = shared_computing.proj_other

        if is_adv_mode:
            opt = self._opt_adv
        else:
            opt = self._opt

        opt.zero_grad()

        def get_loss(use_adv):
            ctx = self.proj.ForwardCtx(
                use_adv=use_adv,
                adv_block_src=is_adv_mode,
                adv_block_self=not is_adv_mode,
            )
            proj_target_in = self.proj(proj_target, ctx=ctx)
            proj_other_out = proj_other - self.proj(proj_other, ctx=ctx)
            proj_recon = proj_target_in + proj_other_out

            recon_logits = shared_computing.logits_bound(
                self._split_concated_proj(proj_recon))
            if seq_pos_mask is not None:
                recon_logits = recon_logits[seq_pos_mask]
            flat = lambda x: x.reshape(-1, x.shape[-1])
            loss = F.cross_entropy(flat(recon_logits),
                                   flat(shared_computing.orig_probs))

            return loss

        if (not is_adv_mode and self.proj.pmat_adv is not None
                and not self._adv_solving_mode.incr_loss):
            # now training our original projection while adv decr loss; our goal
            # is to make the projection better and adv worse

            alpha = 1.5  # the relative importance of loss gap
            loss_proj = get_loss(False)
            loss_adv = get_loss(True)
            loss = loss_proj * (alpha + 1) - loss_adv * alpha
        else:
            loss = get_loss(self.proj.pmat_adv is not None)

        if not is_adv_mode:
            loss = self.proj.pmat.extra_loss(loss)

        neg_loss = is_adv_mode and self._adv_solving_mode.incr_loss
        if neg_loss:
            loss = -loss

        loss.backward()
        opt.step()
        ret = loss.item()
        if neg_loss:
            ret = -ret
        return ret

    @torch.no_grad()
    def replace_with_proj(
            self, c: rc.Circuit,
            updater: typing.Callable[[rc.Circuit], rc.Circuit],
            name_prefix: str = 'proj') -> CircuitWithProjection:
        """replace the matched paths with the solved projection

        :param updater: function to replace the input of a circuit with other
            inputs (i.e., to provide the residual of the projection)
        :param use_adv: whether to use the projection matrix perturbed by the
            adversary
        """
        proj = self.proj.eval()
        if self._adv_solving_mode.replace_by_adv:
            assert proj.pmat_adv is not None
            pmat = proj.pmat_adv
        else:
            pmat = proj.pmat
        p_u_sym = rc.Array(pmat(), name=f'{name_prefix}_proj_u')
        p_c_sym = rc.Array(proj.pcenter.clone(),
                           name=f'{name_prefix}_proj_c')

        def proj_decompose(x: rc.Circuit):
            spec = ''
            # rc does not support ... in einsum spec
            for i in range(x.ndim - 1):
                spec += chr(ord('a') + i)
                assert spec[-1] < 'h'
            p = (
                rc.Einsum.from_einsum_string(
                    f'{spec}h,ho->{spec}o',
                    x.add(p_c_sym.mul_scalar(-1)),
                    p_u_sym).
                add(p_c_sym))
            return p, x.add(p.mul_scalar(-1))

        nodes_name2node = {i.name: i for i in c.get(self._matcher)}
        assert len(nodes_name2node) == len(self._sym_proj_srcs)
        nodes = [nodes_name2node[i.name] for i in self._sym_proj_srcs]
        for i, j in zip(nodes, self._sym_proj_srcs):
            assert i.shape == j.shape or i.shape == j.shape[1:], (
                f'{i.shape=} {j.shape=}'
            )

        proj_inp = rc.Concat(*nodes, axis=-1,
                             name=f'{name_prefix}_inp_target')
        proj_target, _ = proj_decompose(proj_inp)
        _, proj_other = proj_decompose(
            updater(proj_inp.rename(f'{name_prefix}_inp_other')))
        recon = (proj_target.rename(f'{name_prefix}_target').
                 add(proj_other.rename(f'{name_prefix}_other'),
                     name=f'{name_prefix}_recon'))

        begin = 0
        rep_dict = {}

        # rc does not support ... in index
        idx_prefix = I[:] * (nodes[0].ndim - 1)
        for node in nodes:
            sub = recon.index(idx_prefix + I[begin:begin+node.shape[-1]],
                              name=f'{node.name}_recon')
            rep_dict[node] = sub
            begin += node.shape[-1]
        assert begin == recon.shape[-1]

        cnew = c.update(self._matcher, rep_dict.__getitem__)
        assert len(cnew.get(proj_inp.name)) == 1, (
            f'multiple nodes have the name {proj_inp.name}'
        )
        return CircuitWithProjection(cnew, self.proj, proj_inp.name)
