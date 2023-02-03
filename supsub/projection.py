"""tools for orthogonal projection"""

from .utils import get_default_device, torch_as_npy

import scipy.linalg
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function

import typing
import math

@torch.no_grad()
def stable_det_imm(x: torch.Tensor) -> float:
    """try to use a stable method to compute the det when the abs value is not
    too large; imm means immediate number since we return a float instead of a
    pytorch tensor"""
    # det may have very poor stability:
    # https://remix-lpn3709.slack.com/archives/C04FTLKCJSG/p1674260661688979
    sd = torch.linalg.slogdet(x)
    return sd.sign.item() * math.exp(sd.logabsdet.item())

class OrthogonalBase(nn.Module):
    """a random base matrix to be used as the initial base; note that "base"
    means basis in a vector space, not the base in the sense of a base class."""

    dim: int
    matrix: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer('matrix',
                             torch.zeros((dim, dim), dtype=torch.float32))
        self.reset_random()

    def forward(self):
        return self.matrix

    @torch.no_grad()
    def assign_mat(self, u: torch.Tensor) -> "Self":
        """assign a precomputed matrix; we do not check if it is orthogonal"""
        self.matrix.copy_(u)

    @torch.no_grad()
    def reset_random(self) -> "Self":
        """reset the base to be a new sample drawn from SO(n)"""

        # We want a random matrix where the probability measure is invariant
        # under translations (in math jargons, sampling from a distribution
        # according to the Haar measure). Or, we would like to sample X
        # according to a distribution p() such that for an arbitrary orthogonal
        # matrix M, p(MX) = p(X).
        #
        # A nice tutorial:
        # https://www.math.ias.edu/files/wam/Haar_notes-revised.pdf

        dim = self.dim

        # Ensuring SO() by rejection sampling to be safe (not sure if negating
        # the column of a matrix from O()/SO() changes the distribution)
        while True:
            # use float64 for better accuracy
            x0 = torch.empty((dim, dim), dtype=torch.float64,
                             device=self.matrix.device).normal_()

            q, r = torch.linalg.qr(x0)

            # ensure r >= 0 for Gram-Schmidt process
            d = r.diagonal()
            assert d.ndim == 1
            if abs(torch.prod(d).item()) < 1e-4:
                # unlucky to get a singular matrix; try again
                continue
            ds = d.sign()
            q *= ds.unsqueeze(0)
            r *= ds.unsqueeze(1)
            assert torch.all(r.diagonal() > 0)
            assert torch.allclose(q @ r, x0)

            det = stable_det_imm(q)
            if det < 0:
                # we should be luckier next time!
                continue

            assert abs(det - 1) < 1e-4
            self.matrix.copy_(q)
            return self


class SkewSymmetricMatrix(nn.Module):
    """an skew-symmetric matrix to parameterize SO(n); initialized as zeros"""

    matrix: nn.Parameter

    def __init__(self, dim: int):
        super().__init__()
        self.matrix = nn.Parameter(torch.zeros((dim, dim), dtype=torch.float32))

    def forward(self):
        x = self.matrix
        # only take a lower triangle to reduce the DOF in optimization
        x = torch.tril(x, -1)
        return x - x.T

    @torch.no_grad()
    def reset_zero(self):
        """reset to all-zero matrix"""
        self.matrix.zero_()

    @property
    def dim(self) -> int:
        return self.matrix.shape[0]


class OrthogonalMatrixBase(nn.Module):
    """base class for parameterized orthogonal matrices"""

    def assign_mat(self, u: torch.Tensor):
        """reset the parameters and assign a new matrix as the next forward()
        result"""
        raise NotImplementedError()


class OrthogonalMatrixCayley(OrthogonalMatrixBase):
    """orthogonal matrix from Cayley transform"""
    base: OrthogonalBase
    skew_sym: SkewSymmetricMatrix
    I: torch.Tensor

    def __init__(self, dim: int):
        super().__init__()
        self.base = OrthogonalBase(dim)
        self.skew_sym = SkewSymmetricMatrix(dim)
        self.register_buffer('I', torch.eye(dim))

    def forward(self):
        """Returns an orthogonal matrix generated from self.matrix using the
        Cayley transform"""
        x = self.skew_sym()
        I = self.I
        return self.base() @ torch.linalg.solve(I - x, I + x)

    @torch.no_grad()
    def assign_mat(self, u: torch.Tensor) -> "Self":
        """assign a precomputed orthogonal matrix to be used as the result of
        forward"""
        self.base.assign_mat(u)
        self.skew_sym.reset_zero()
        return self


class OrthogonalMatrixExp(OrthogonalMatrixBase):
    """orthogonal matrix from matrix exponential"""
    base: OrthogonalBase
    skew_sym: SkewSymmetricMatrix

    def __init__(self, dim: int):
        super().__init__()
        self.base = OrthogonalBase(dim)
        self.skew_sym = SkewSymmetricMatrix(dim)

    def forward(self):
        """Returns an orthogonal matrix generated from self.matrix using the
        matrix exponential"""
        # this works because XY = YX -> exp(X+Y)=exp(X)exp(Y), and exp(0)=I
        x = self.skew_sym()
        return self.base() @ torch.linalg.matrix_exp(x)

    @torch.no_grad()
    def assign_mat(self, u: torch.Tensor) -> "Self":
        """assign a precomputed orthogonal matrix to be used as the result of
        forward"""
        self.base.assign_mat(u)
        self.skew_sym.reset_zero()
        return self


class OrthogonalMatrixGeotorch(OrthogonalMatrixBase):
    """orthogonal matrix from geotroch"""
    # note: I believe the above code reproduces geotorch

    def __init__(self, dim: int):
        # import locally to reduce our global dependencies
        import geotorch
        super().__init__()
        self.dim = dim
        self.matrix = nn.Linear(dim, dim, bias=False)
        geotorch.orthogonal(self.matrix, 'weight')

    def forward(self):
        """Returns an orthogonal matrix generated from self.matrix using
        geotorch"""
        return self.matrix.weight

    @torch.no_grad()
    def assign_mat(self, u: torch.Tensor):
        """assign a precomputed orthogonal matrix to be used as the result of
        forward"""
        self.matrix.weight = u


class AffineMatrix(OrthogonalMatrixBase):
    """an arbitrary matrix for low dimensional affine transformations; actually
    not an orthogonal matrix"""
    dim: int
    matrix: nn.Parameter

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # initialize as orth
        self.matrix = nn.Parameter(OrthogonalBase(dim)().detach_())

    def forward(self):
        return self.matrix


def make_orthogonal_matrix(
        dim: int,
        factorization_type: typing.Optional[str] = None) -> OrthogonalMatrixBase:
    """
    Factory for generating orthogonal matrices from unconstrained matrices.
    Caller can specify which factorization method to use.

    :param dim: dimension of the (square) matrix to generate
    :param factorization_type: one of 'cayley', 'exp', or 'geotorch'; default is
        cayley
    """
    if factorization_type is None:
        # only matrix exponential creates a one-to-one mapping between
        # skew-symmetric matrices and SO(n)
        factorization_type = 'exp'

    match factorization_type.lower():
        case 'cayley':
            return OrthogonalMatrixCayley(dim)
        case 'exp':
            return OrthogonalMatrixExp(dim)
        case 'geotorch':
            return OrthogonalMatrixGeotorch(dim)
        case 'affine':
            return AffineMatrix(dim)
        case _:
            raise NotImplementedError(
                f'unknown factorization type {factorization_type}')


class ProjectionMatrixBase(nn.Module):
    """base class of low-rank projection matrices"""

    u: OrthogonalMatrixBase
    """the underlying full-rank projection matrix; each row is a basis"""

    dim: int
    """total dimension of the input space"""

    proj_dim: int
    """dimension of the internal projection space"""

    def __init__(self, dim: int, proj_dim: int, factorization_type):
        super().__init__()
        if proj_dim is not None:
            assert 0 < proj_dim <= dim, f'{proj_dim=} {dim=}'
            self.proj_dim = proj_dim
        self.dim = dim
        self.u = make_orthogonal_matrix(dim, factorization_type)

    def extra_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """add extra terms to the loss function; the default is identity"""
        return loss

    @torch.no_grad()
    def _format_repr(self) -> str:
        det = stable_det_imm(self.u())
        return f'type={self.u.__class__.__name__}, det(u)={det:.2f}'

    def make_diag(self) -> torch.Tensor:
        """compute the diagonal matrix for projection; shape is (n, )"""
        raise NotImplementedError()

    def forward(self) -> torch.Tensor:
        """compute the complete projection matrix to be right-multiplied with
        data (projected features should be ``data @ self()`` where
        ``data.shape`` is (batch, dim)"""
        u = self.u()
        return (u.T * self.make_diag().unsqueeze(0)) @ u


class ProjectionMatrix(ProjectionMatrixBase):
    """Projection matrix generation from an orthogonal matrix """

    d: torch.Tensor
    """the diagonal matrix (n, n) shape"""

    def __init__(self, dim: int, proj_dim: int,
                 factorization_type: typing.Optional[str] = None):
        super().__init__(dim, proj_dim, factorization_type)
        self.register_buffer(
            'd', torch.cat([torch.ones(proj_dim), torch.zeros(dim - proj_dim)])
        )

    def make_diag(self) -> torch.Tensor:
        return self.d

    def __repr__(self):
        return f'ProjMat({self._format_repr()})'


class Binarize01WeightNoScaleFn(Function):
    """weight binarization. The code comes from the BinMask part in the paper
    EEVBNN, please cite it if you use this code:
    https://arxiv.org/abs/2005.03597
    """
    weight_decay = 1e-4

    @classmethod
    def forward(cls, ctx, inp, weight_decay=None):
        v = (inp >= 0).to(inp.dtype)
        if weight_decay is None:
            ctx.save_for_backward(v)
        else:
            ctx.save_for_backward(v, weight_decay)
        return v

    @classmethod
    def backward(cls, ctx, g_out):
        if len(ctx.saved_tensors) == 1:
            out, = ctx.saved_tensors
            weight_decay = cls.weight_decay
        else:
            out, weight_decay = ctx.saved_tensors
        grad = (out * weight_decay).add_(g_out)
        if len(ctx.saved_tensors) == 1:
            return grad
        return grad, None


class AdaptiveProjectionMatrixBin(ProjectionMatrixBase):
    """use binarization for adaptive projection"""

    diag_real: nn.Parameter
    """the diagonal entries with real values"""

    weight_decay: float

    def __init__(self, dim: int, proj_dim: int = -1,
                 factorization_type: typing.Optional[str] = None,
                 weight_decay: float=1e-4):
        super().__init__(dim, None, factorization_type)
        self.diag_real = nn.Parameter(
            torch.empty((dim, ), dtype=torch.float32).normal_(std=1e-2).abs_()
        )
        self.register_buffer('weight_decay',
                             torch.tensor(weight_decay, dtype=torch.float32))

    def make_diag(self) -> torch.Tensor:
        return self.diag_bin

    @property
    def diag_bin(self):
        """diagonal entries with binary values"""
        return Binarize01WeightNoScaleFn.apply(self.diag_real, self.weight_decay)

    @property
    def proj_dim(self) -> int:
        """the current dimension of the projection space"""
        return torch.count_nonzero(self.diag_bin).item()

    def __repr__(self):
        wavg = self.diag_real[self.diag_real > 0].mean()
        return (f'AdaProjBin({self._format_repr()},'
                f' dim={self.proj_dim}, {wavg=:.3f})')


class AdaptiveProjectionMatrixL1(ProjectionMatrixBase):
    """add L1 decay to the required zero dimensions"""

    diag_hard: torch.Tensor

    diag_soft: nn.Parameter
    """the diagonal entries with real values"""

    diag_soft_init: float = 0.1

    weight_decay: float

    def __init__(self, dim: int, proj_dim: int,
                 factorization_type: typing.Optional[str] = None,
                 weight_decay: float=1e-4):
        super().__init__(dim, proj_dim, factorization_type)
        self.register_buffer('diag_hard',
                             torch.ones(proj_dim, dtype=torch.float32))
        self.diag_soft = nn.Parameter(
            torch.empty(dim - proj_dim, dtype=torch.float32).
            fill_(self.diag_soft_init).
            detach_())
        self.weight_decay = weight_decay

    @property
    def diag_soft_val(self) -> torch.Tensor:
        """the soft diagonal entries after some transform"""
        return self.diag_soft.abs() / self.diag_soft_init

    def make_diag(self):
        if self.training:
            dsoft = self.diag_soft_val
        else:
            dsoft = torch.zeros_like(self.diag_soft)
        return torch.cat([self.diag_hard, dsoft], dim=0)

    def extra_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss + self.diag_soft_val.sum() * self.weight_decay

    def __repr__(self):
        wavg = self.diag_soft_val.mean()
        return (f'AdaProjL1({self._format_repr()},'
                f' dim={self.proj_dim}, {wavg=:.3f})')


class IdentityMatrix(OrthogonalMatrixBase):
    """a non-trainable I of constant size"""

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer('I', torch.eye(dim, dtype=torch.float32))

    def forward(self):
        return self.I


class AdversarialProjectionMatrix(nn.Module):
    """an adversarial projection matrix that removes some dimensions and adds
    some new dimensions"""

    src_proj: ProjectionMatrixBase

    num_rm: int
    """number of dimensions in the projection subspace to remove"""

    num_add: int
    """number of dimensions to be added to the projection subspace"""

    shuf_in: OrthogonalMatrixBase
    """bases to shuffle the projection subspace"""

    shuf_out: OrthogonalMatrixBase
    """bases to shuffle the non-projection subspace"""

    add_mask: torch.Tensor
    """1 for new diag entries to add"""

    remove_mask: torch.Tensor
    """0 for diag entries to remove"""

    @torch.no_grad()
    def __init__(self, src_proj: ProjectionMatrixBase,
                 num_rm: int, num_add: int, factorization_type):
        super().__init__()
        assert (0 <= min(num_rm, num_add) and (num_rm or num_add) and
                num_rm < src_proj.proj_dim and
                num_add < src_proj.dim - src_proj.proj_dim)
        self.src_proj = src_proj
        self.num_rm = num_rm
        self.num_add = num_add
        proj_dim = src_proj.proj_dim
        noproj_dim = src_proj.dim - src_proj.proj_dim
        assert proj_dim > 0 and noproj_dim > 0
        if num_rm:
            self.shuf_in = make_orthogonal_matrix(proj_dim, factorization_type)
        else:
            self.shuf_in = IdentityMatrix(proj_dim)

        if num_add:
            self.shuf_out = make_orthogonal_matrix(
                noproj_dim, factorization_type)
        else:
            self.shuf_out = IdentityMatrix(noproj_dim)

        remove_mask = torch.ones(src_proj.dim, dtype=torch.float32)
        if num_rm:
            remove_mask[proj_dim-num_rm:proj_dim] = 0

        add_mask = torch.zeros_like(remove_mask)
        if num_add:
            add_mask[-num_add:] = 1

        self.register_buffer('remove_mask', remove_mask)
        self.register_buffer('add_mask', add_mask)

    def forward(self, block_src_grad: bool=False, block_self_grad: bool=False):
        """
        :param block_src_grad: if set to True, will block the gradient into
            ``src_proj``
        :param block_self_grad: if set to True, will block the gradient into
            parameters of this layer
        """
        src_u = self.src_proj.u()
        src_d = self.src_proj.make_diag()
        if block_src_grad:
            src_u = src_u.detach()
            src_d = src_d.detach()

        shuf = torch.block_diag(self.shuf_in(), self.shuf_out())
        if block_self_grad:
            shuf = shuf.detach()

        # note that basis in u are row vectors
        u = shuf @  src_u
        diag = torch.maximum(src_d * self.remove_mask, self.add_mask)
        return (u.T * diag.unsqueeze(0)) @ u

    def __repr__(self):
        return f'{self.__class__.__name__}(-{self.num_rm}+{self.num_add})'


class ProjectionLayer(nn.Module):
    """the projection layer that includes a projection matrix and an offset"""

    pmat: ProjectionMatrixBase
    pmat_adv: typing.Optional[AdversarialProjectionMatrix]
    """the adversarial projection matrix"""
    pcenter: nn.Parameter

    class ForwardCtx:
        _p = None
        _use_adv: bool

        def __init__(self, use_adv: bool = False,
                     adv_block_src: bool = False,
                     adv_block_self: bool = False):
            self._use_adv = use_adv
            self._adv_block_src = adv_block_src
            self._adv_block_self = adv_block_self

        def get_pmat(self, pl):
            if self._p is None:
                if self._use_adv:
                    assert pl.pmat_adv is not None
                    self._p = pl.pmat_adv(block_src_grad=self._adv_block_src,
                                          block_self_grad=self._adv_block_self)
                else:
                    self._p = pl.pmat()
            return self._p

    def __init__(self, dim, proj_dim, factorization_type=None,
                 use_adaptive: typing.Optional[str]=None,
                 adaptive_decay: float=1e-4,
                 adv_proj_dim: typing.Union[int, tuple[int, int]] = 0):
        """:param use_adaptive: adaptive projection method: l1 or bin
        :param adv_proj_dim: number of dimensions to be changed (removed and
            then added) by the adversary, or the individual numbers of
            dimensions to remove and add. A value of zero disables the
            adversary.
        """
        super().__init__()

        def make_proj(pdim) -> ProjectionMatrixBase:
            if use_adaptive == 'l1':
                return AdaptiveProjectionMatrixL1(
                    dim, pdim, factorization_type, weight_decay=adaptive_decay)
            if use_adaptive == 'bin':
                return AdaptiveProjectionMatrixBin(
                    dim, pdim, factorization_type, weight_decay=adaptive_decay)
            if not use_adaptive:
                return ProjectionMatrix(dim, pdim, factorization_type)
            raise NotImplementedError(
                f'unhandled adaptive proj type {use_adaptive}')

        self.pmat = make_proj(proj_dim)
        if adv_proj_dim:
            if isinstance(adv_proj_dim, int):
                num_rm = num_add = adv_proj_dim
            else:
                num_rm, num_add = adv_proj_dim
            self.pmat_adv = AdversarialProjectionMatrix(
                self.pmat, num_rm, num_add, factorization_type)
        else:
            self.pmat_adv = None

        self.pcenter = nn.Parameter(torch.zeros((dim, ), dtype=torch.float32))

    def __repr__(self):
        ret = (f'{self.__class__.__name__}({self.dim}->{self.proj_dim},'
               f' {self.pmat}')
        if (m := self.pmat_adv) is not None:
            ret += f'; adv: {m}'
        return ret + ')'

    def _canonize_input(self, x: torch.Tensor) -> torch.Tensor:
        """check shape of x, and reshape to 2d"""
        assert x.ndim >= 2 and x.shape[-1] == self.dim, (
            f'{x.shape=} {self.dim=}')
        return x.reshape(-1, self.dim)

    @torch.no_grad()
    def reset_as_pca(self, x: torch.Tensor, bias_only=False):
        """reset the parameters to be the PCA of given features

        :param x: the features in ``(..., dim)`` shape
        :param bias_only: if true, will only set the bias but not the projection
            matrix
        """
        x = self._canonize_input(x)
        ex = x.mean(dim=0, keepdim=True)
        self.pcenter.copy_(ex.squeeze(0))
        if bias_only:
            return
        xz = x - ex
        cov = xz.T @ xz
        cov /= x.shape[0]
        eig_val, eig_vec = torch.linalg.eigh(cov)
        if stable_det_imm(eig_vec) < 0:
            eig_vec[:, 0] = -eig_vec[:, 0]
        self.pmat.u.assign_mat(torch.flip(eig_vec, dims=(1, )).T)

    @classmethod
    def make_identity(cls, dim: int):
        """create an identity projection; mostly used for debug"""
        ret = cls(dim, dim)
        ret.pmat.u.assign_mat(torch.eye(dim))
        return ret

    @classmethod
    def make_identity_like(cls, other: 'ProjectionLayer'):
        return cls.make_identity(other.dim).to(other.pcenter.device)

    @property
    def dim(self) -> int:
        """feature space dimension"""
        return self.pmat.dim

    @property
    def proj_dim(self) -> int:
        """dimension of the internal projection space"""
        return self.pmat.proj_dim

    def forward(self, x: torch.Tensor,
                ctx: typing.Optional[ForwardCtx]=None) -> torch.Tensor:
        """compute the projection of ``x`` represented in the original
        high-dimensional space

        :param x: of shape ``(..., dim)``
        :param ctx: a context object serving two objectives:
            1. Avoiding repeated computation if you want to forward the same
                ProjectionLayer on multiple inputs
            2. Passing additional params for forward
        :return: a tensor with shape ``(..., dim)``
        """
        x_shape = x.shape
        x = self._canonize_input(x)
        if ctx is None:
            ctx = self.ForwardCtx()
        p = ctx.get_pmat(self)
        c = self.pcenter
        return ((x - c) @ p + c).reshape(x_shape)

    def get_projection(self, x: torch.Tensor) -> torch.Tensor:
        """get the projection result of ``x`` in the internal space

        :param x: of shape ``(..., dim)``
        :return: a tensor with shape ``(..., proj_dim)``
        """
        self._canonize_input(x) # just check input shape here
        u = self.pmat.u()
        p = torch.einsum('...i,oi->...o', x - self.pcenter, u)
        diag = self.pmat.make_diag()
        return p[:, diag != 0]


@torch.no_grad()
def run_tests():
    # note: since this module depends on other modules in this package, use
    # command line `python -m supsub.projection` to run this test. Interactive
    # notebooks may not work here.
    import time
    if torch.cuda.is_available():
        device = 'cuda'
        sync = torch.cuda.synchronize
    else:
        device = 'cpu'
        sync = lambda: None
    niter = 10
    for dim in [8, 768]:
        I = torch.eye(dim, device=device, dtype=torch.float32)
        tgt = OrthogonalMatrixCayley(dim).to(device)()
        for fac in ['cayley', 'exp', 'geotorch']:
            m = make_orthogonal_matrix(dim, fac).to(device)
            m()
            sync()
            t0 = time.time()
            for i in range(niter):
                y = m()
            sync()
            t1 = time.time()
            r = y.T @ y
            assert (diff := (r - I).abs().max()) < 1e-4, (r, diff)
            print(f'{fac}@{dim}: time={(t1-t0)/niter*1e3:.2f}ms/forward')
            m.assign_mat(tgt)
            torch.testing.assert_close(m(), tgt, atol=1e-4, rtol=1e-4)

if __name__ == '__main__':
    run_tests()
