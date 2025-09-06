import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros, constant


class MVLayerNorm(nn.Module):
    """
    Equivariant layer normalization of multivectors.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
    """

    algebra: object

    def __call__(self, input):
        """Compute norm of the input and normalize the input as input / norm.
        The norm is computed w.r.t the extended quadratic form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            output (jnp.ndarray): normalized multivector of shape (..., 2**algebra.dim).
        """
        norms = self.algebra.norm(input).mean(axis=1, keepdims=True)
        outputs = input / (norms + 1e-6)
        return outputs


class GradeNorm(nn.Module):
    """
    Equivariant per-grade normalization of multivectors with learnable factors.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
    """

    algebra: object

    @nn.compact
    def __call__(self, input):
        """Compute norms of the input grades and normalize the input as input / (factor * norms).
        The norms are computed w.r.t the extended quadratic form (see Eq. 2 of https://arxiv.org/abs/2305.11141).

        Args:
            input (jnp.ndarray): multivector of shape (..., 2**algebra.dim).

        Returns:
            output (jnp.ndarray): normalized multivector of shape (..., 2**algebra.dim).
        """
        norms = jnp.concatenate(self.algebra.norms(input), axis=-1)

        factor = self.param(
            "factor", zeros, (1, norms.shape[1], self.algebra.n_subspaces)
        )
        factor = jax.lax.broadcast_in_dim(
            factor, norms.shape, (0, 1, len(norms.shape) - 1)
        )
        norms = jnp.repeat(
            jax.nn.sigmoid(factor) * (norms - 1) + 1, self.algebra.subspaces, axis=-1
        )
        return input / (norms + 1e-6)
    
# TODO: THIS IS THE CONDITION GRADE NORM THAT I WAS TRYING
# TO USE TO NORMALISE THE CONDITION SO THAT THE VALUES DON'T EXPLODE TO NAN IN THE 2D MAXWELL SIM.
class ConditionGradeNorm(nn.Module):
    """Grade-wise scaling of the *tanh-gated* condition by a learnable factor.

    The scaling parameter α_g is learned per grade and is **constant w.r.t. the
    batch**, so we create it once in `setup` with a **static** shape that only
    depends on the known number of channels.
    """

    algebra: object
    num_channels: int  # <<< static, passed from parent module

    def setup(self):
        # One learnable multiplier per (channel, grade)
        n_sub = int(self.algebra.n_subspaces)
        self.grade_mult = self.param(
            "grade_mult", constant(1.0), (1, int(self.num_channels), n_sub)
        )

    def __call__(self, input, relpos_norm):
        """Scale each grade by tanh(input) * (α_g * ∥relpos∥).

        Parameters
        ----------
        input:       (P, C, 2**dim) tensor – condition replicated over kernel points
        relpos_norm: (P,) **or** (P, n_subspaces) – norm per grid point
        """

        # Non-linearity first
        gated = jax.nn.tanh(input)

        # Broadcast learnable multipliers over positions
        grade_mult = jnp.broadcast_to(self.grade_mult, (gated.shape[0],) + self.grade_mult.shape[1:])

        # Bring relpos_norm to shape (P, 1, n_subspaces)
        if relpos_norm.ndim == 1:
            relpos_norm = relpos_norm[:, None]
        relpos_norm = relpos_norm.reshape(gated.shape[0], 1, -1)

        scale_per_grade = grade_mult * relpos_norm  # (P, C, n_subspaces)

        # Replicate per-grade factor over blades within each grade
        scale = jnp.repeat(scale_per_grade, self.algebra.subspaces, axis=-1)

        return gated * scale
