import jax.numpy as jnp
import jax
from flax import linen as nn
from jax.nn.initializers import ones

from ..core.cayley import WeightedCayley
from .shell import ComposedScalarShell
from .network import KernelNetwork
from algebra.cliffordalgebra import CliffordAlgebra
from .kernel import CliffordSteerableKernel, generate_kernel_grid, get_init_factor


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
    return jax.lax.conv(k1, k2, window_strides=(1, 1), padding="SAME")

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
    Convolves 2 kernels composing them into a single kernel.
    Returns: Composed kernel of shape (c_out * algebra.n_blades, c_in * algebra.n_blades, X_1, ..., X_dim).
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
        self.kernel_params = (
            self.algebra,
            self.c_in,
            self.c_out,
            self.kernel_size,
            self.num_layers,
            self.hidden_dim,
            self.bias_dims,
            self.product_paths_sum,
        )

    @nn.compact
    def __call__(self):
        """
        Evaluate the steerable implicit kernel.
        Returns: Composed kernel of shape (c_out * algebra.n_blades, c_in * algebra.n_blades, X_1, ..., X_dim).
        """

        # Generate individual kernels
        k1, rel_pos, factor = CliffordSteerableKernel(*self.kernel_params)()
        k2, _, _ = CliffordSteerableKernel(*self.kernel_params)()

        # Convolve the kernels to get the composed kernel
        k = conv_kernel(self.algebra, k1, k2)

        # Compute the shell for the composed kernel
        shell_comp = ComposedScalarShell(self.algebra, self.c_in, self.c_out)(rel_pos) #output shape: (N, c_out, c_in, 2**algebra.dim, 2**algebra.dim)
        shell_comp = shell_comp.transpose(1, 2, 3, 4, 0).reshape(self.c_out * 2 ** self.algebra.dim, self.c_in * 2 ** self.algebra.dim, *(self.algebra.dim * [self.kernel_size]))


        # Apply the scalar shell to the composed kernel
        K = k * shell_comp * factor


        return K
