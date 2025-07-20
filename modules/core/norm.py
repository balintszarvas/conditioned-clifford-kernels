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
        print("norms in grade norm", norms)
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
    """Grade–wise scaling of the *tanh-gated* condition by a learnable multiple
    of the norm of the relative-position vector.

    The operation is
        out = tanh(input) * (α_g * ∥relpos∥)
    where α_g is a learned positive scalar for every grade g.  Because ∥relpos∥
    is invariant under O(p,q) and the scaling factor is per-grade (scalar), the
    transformation is O(p,q)-equivariant and grade-preserving.
    """

    algebra: object  # instance of CliffordAlgebra passed from the parent module

    @nn.compact
    def __call__(self, input, relpos_norm):
        """Apply tanh and scale grade-wise.

        Args
        ----
        input:        jnp.ndarray with shape (P, C, 2**dim) – condition repeated
                       over the kernel grid positions.
        relpos_norm:  jnp.ndarray with shape (P, 1) or (P,) giving ∥relpos∥ for
                       each position of the kernel grid.

        Returns
        -------
        output:      jnp.ndarray of the same shape as *input*.
        """

        relpos_norm = jnp.asarray(relpos_norm)  # ensure JAX array

        # 1. Non-linearity
        gated = jax.nn.tanh(input)

        # 2. Per-grade learnable multipliers α_g ≥ 0
        n_channels = gated.shape[1]
        grade_mult = self.param(
            "grade_mult", constant(1.0), (1, n_channels, self.algebra.n_subspaces)
        )

        # Broadcast α_g over positions (P) and make shape (P, C, n_subspaces)
        grade_mult = jnp.broadcast_to(grade_mult, (gated.shape[0],) + grade_mult.shape[1:])

        # 3. Broadcast relpos_norm (shape P,n_subspaces) over channels
        #    Expect relpos_norm.shape == (P, n_subspaces)
        if relpos_norm.ndim == 1:
            relpos_norm = relpos_norm[:, None]  # (P,1)

        # reshape to (P,1,n_subspaces) then broadcast multiply
        relpos_norm = relpos_norm.reshape(gated.shape[0], 1, -1)
        scale_per_grade = grade_mult * relpos_norm  # (P, C, n_subspaces)

        # 4. Replicate the per-grade factor over the blades inside each grade
        scale = jnp.repeat(scale_per_grade, self.algebra.subspaces, axis=-1)  # (P,C,2**dim)

        return gated * scale
