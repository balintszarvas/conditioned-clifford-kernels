import jax.numpy as jnp
import jax
from flax import linen as nn
from jax.nn.initializers import ones

from ..core.cayley import WeightedCayley
from .shell import ScalarShell, compute_scalar_shell
from .network import KernelNetwork
from algebra.cliffordalgebra import CliffordAlgebra
from .kernel import CliffordSteerableKernel, generate_kernel_grid, get_init_factor


class CondCliffordSteerableKernel(nn.Module):
    """
    Conditioned Clifford-steerable kernel (see Section 3, Appendix A for details).
    It consists of two components:
        1. A kernel network that generates a stack of c_in * c_out multivectors using an O(p,q)-equivariant CEGNN (Ruhe et al., 2023)
            k: R^p,q -> Cl^(c_out * c_in)
        2. A kernel head that converts the stack of multivectors into a kernel.
            K: Cl^[c_out x c_in] -> Hom_vec(Cl^c_in, Cl^c_out)

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    num_layers: int
    hidden_dim: int
    bias_dims: tuple
    product_paths_sum: int

    def setup(self):
        self.rel_pos = generate_kernel_grid(self.kernel_size, self.algebra.dim)[
            :, jnp.newaxis, :
        ]
        self.factor = get_init_factor(self.algebra, self.kernel_size)
        self.rel_pos_sigma = self.param("rel_pos_sigma", ones, (1, 1, 1))

    @nn.compact
    def __call__(self, condition):
        """
        Evaluate the steerable implicit kernel.

        Inputs:
            condition (jnp.ndarray): The condition multivector of shape (c_in, X_1, ..., X_dim, 2**algebra.dim).

        Returns:
            The output kernel of shape (c_out * algebra.n_blades, c_in * algebra.n_blades, X_1, ..., X_dim).
        """
        # Weighted Cayley
        weighted_cayley = WeightedCayley(
            self.algebra, self.c_in, self.c_out, self.product_paths_sum
        )()

        # Compute scalars
        scalar = compute_scalar_shell(self.algebra, self.rel_pos, self.rel_pos_sigma)

        # Embed scalar and vector into a multivector
        x = self.algebra.embed_grade(scalar, 0) + self.algebra.embed_grade(self.rel_pos, 1) # [P,1,4]

        # Broadcast condition and concatenate with relative positions
        condition = jnp.repeat(condition[jnp.newaxis], len(self.rel_pos), axis=0) # [C,4] -> [P,C,4]

        x = jnp.concatenate([x, condition], axis=1) # [P,1,4] x [P,C,4] -> [P,C+1,4]

        # Evaluate kernel network
        k = KernelNetwork(
            self.algebra,
            self.c_in + 1,
            self.c_in,
            self.c_out,
            self.num_layers,
            self.hidden_dim,
            self.bias_dims,
            self.product_paths_sum,
        )(x)

        # Reshape to kernel mask
        k = k.reshape(-1, self.c_out, self.c_in, self.algebra.n_blades)

        # Compute kernel mask
        shell = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2**self.algebra.dim
        )

        # Mask kernel
        k = k * shell * self.factor

        # Kernel head: partial weighted geometric product
        K = jnp.einsum("noik,oiklm->olimn", k, weighted_cayley)

        # Handle N-dimensional reshaping
        spatial_dims = self.algebra.dim * [self.kernel_size]
        K = K.reshape(
            self.c_out * self.algebra.n_blades,
            self.c_in * self.algebra.n_blades,
            *spatial_dims
        )

        return K