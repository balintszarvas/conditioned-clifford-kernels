import jax.numpy as jnp
import jax
from flax import linen as nn
from jax.nn.initializers import ones

from modules.core.cayley import WeightedCayley
from modules.conv.shell import ScalarShell, compute_scalar_shell
from modules.conv.network import KernelNetwork
from algebra.cliffordalgebra import CliffordAlgebra
from modules.conv.kernel import CliffordSteerableKernel, generate_kernel_grid, get_init_factor


def reshape_mv_tensor(algebra, tensor):
    A, B, *dims = tensor.shape
    M = A // algebra.n_blades
    N = B // algebra.n_blades
    return tensor.reshape(M,algebra.n_blades,N,algebra.n_blades,*dims)

def reshape_back(algebra, tensor):
    M, _, N, _, *dims = tensor.shape
    A = M * algebra.n_blades
    B = N * algebra.n_blades
    return tensor.reshape(A,B,*dims)

def _conv_kernel_11(k1, k2):
    return jax.lax.conv(k1, k2, window_strides=(1,1), padding="SAME")

def _conv_kernel_14(k1, k2):
    return jax.vmap(_conv_kernel_11, in_axes=(0, 0))(k1, k2)

def conv_kernel(algebra, k1, k2):
    k1 = reshape_mv_tensor(algebra, k1).transpose(0, 2, 1, 3, 4, 5)
    k2 = reshape_mv_tensor(algebra, k2).transpose(0, 2, 1, 3, 4, 5)
    k = jax.vmap(_conv_kernel_14, in_axes=(0, 0))(k1, k2)
    k = k.transpose(0, 2, 1, 3, 4, 5)
    k = reshape_back(algebra, k)
    return k

class ComposedCliffordSteerableKernel(nn.Module):
    """
    Extended Clifford-steerable kernel (see Section 3, Appendix A for details).
    Convolves 2 kernels with different kernel sizes and combines them.
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
        """
        Initialize the kernel parameters.
        """

        self.rel_pos = generate_kernel_grid(self.kernel_size, self.algebra.dim)[
            :, jnp.newaxis, :
        ]
        self.factor = get_init_factor(self.algebra, self.kernel_size)
        self.rel_pos_sigma1 = self.param("rel_pos_sigma1", ones, (1, 1, 1))
        self.rel_pos_sigma2 = self.param("rel_pos_sigma2", ones, (1, 1, 1))


        # Compute the new kernel size after convolution
        #self.composed_kernel_size = 2 * self.kernel_size - 1

        # Generate the relative position grid for the composed kernel
        #self.rel_pos_comp = generate_kernel_grid(self.composed_kernel_size, self.algebra.dim)[:, jnp.newaxis, :]

        # Initialize the sigma parameter for the scalar shell of the composed kernel
        #self.comp_factor = get_init_factor(self.algebra, self.composed_kernel_size)
        #self.rel_pos_sigma_comp = self.param("rel_pos_sigma_comp", ones, (1, 1, 1))

    @nn.compact
    def __call__(self):
        """
        Evaluate the steerable implicit kernel.
        Returns: Composed kernel of shape (c_out * algebra.n_blades, c_in * algebra.n_blades, X_1, ..., X_dim).
        """
        
        # Weighted Cayley
        weighted_cayley1 = WeightedCayley(
            self.algebra, self.c_in, self.c_out, self.product_paths_sum
        )()

        weighted_cayley2 = WeightedCayley(
            self.algebra, self.c_in, self.c_out, self.product_paths_sum
        )()

        # Compute scalars
        scalar1 = compute_scalar_shell(self.algebra, self.rel_pos, self.rel_pos_sigma1)
        scalar2 = compute_scalar_shell(self.algebra, self.rel_pos, self.rel_pos_sigma2)

        x1 = self.algebra.embed_grade(scalar1, 0)+ self.algebra.embed_grade(self.rel_pos, 1)
        x2 = self.algebra.embed_grade(scalar2, 0)+ self.algebra.embed_grade(self.rel_pos, 1)

        # Evaluate kernel network
        k1 = KernelNetwork(
            self.algebra,
            self.c_in,
            self.c_out,
            self.num_layers,
            self.hidden_dim,
            self.bias_dims,
            self.product_paths_sum,
        )(x1)

        # Evaluate kernel network
        k2 = KernelNetwork(
            self.algebra,
            self.c_in,
            self.c_out,
            self.num_layers,
            self.hidden_dim,
            self.bias_dims,
            self.product_paths_sum,
        )(x2)

        k1 = k1.reshape(-1, self.c_out, self.c_in, self.algebra.n_blades)
        k2 = k2.reshape(-1, self.c_out, self.c_in, self.algebra.n_blades)

        # Apply the partial weighted geometric product to kernel1
        K1 = jnp.einsum("noik,oiklm->olimn", k1, weighted_cayley1) #output shape: (c_out, algebra.n_blades, c_in, algebra.n_blades, X ** dim)

        # Appy the partial weighted geometric product to kernel2
        K2 = jnp.einsum("noik,oiklm->olimn", k2, weighted_cayley2) #output shape: (c_out, algebra.n_blades, c_in, algebra.n_blades, X ** dim)

        # Reshape K1 to final kernel
        K1 = K1.reshape(
            self.c_out * self.algebra.n_blades,
            self.c_in * self.algebra.n_blades,
            *(self.algebra.dim * [self.kernel_size]),
            )

        K2 = K2.reshape(
            self.c_out * self.algebra.n_blades,
            self.c_in * self.algebra.n_blades,
            *(self.algebra.dim * [self.kernel_size])
            )
        
        # Convolve the kernels to get the composed kernel
        k = conv_kernel(self.algebra, K1, K2) #output shape: (c_out, algebra.n_blades, c_in, algebra.n_blades, X ** dim)


        # Compute the shell for the composed kernel
        shell_comp1 = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2 ** self.algebra.dim
        )

        shell_comp2 = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2 ** self.algebra.dim
        )

        shell_comp3 = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2 ** self.algebra.dim
        )

        shell_comp4 = ScalarShell(self.algebra, self.c_in, self.c_out)(self.rel_pos).reshape(
            -1, self.c_out, self.c_in, 2 ** self.algebra.dim
        )

        shell_comp = jnp.concatenate([shell_comp1, shell_comp2, shell_comp3, shell_comp4], axis=3)


        shell_comp = shell_comp.transpose(1, 2, 3, 0).reshape(self.c_out * 2 ** self.algebra.dim, self.c_in * 2 ** self.algebra.dim, *(self.algebra.dim * [self.kernel_size]))


        # Apply the scalar shell to the composed kernel
        K = k * shell_comp * self.factor

        # Reshape to final kernel
        K = K.reshape(
            self.c_out * self.algebra.n_blades,
            self.c_in * self.algebra.n_blades,
            *(self.algebra.dim * [self.kernel_size])
        )

        return K