import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros

from .kernel import CliffordSteerableKernel
from .ckernel import ComposedCliffordSteerableKernel
from .condkernel import CondCliffordSteerableKernel



class CliffordSteerableConv(nn.Module):
    """
    Clifford-steerable convolution layer. See Section 3, Appendix A for details. Pseudocode is given in Function 3.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        kernel_type (string): The type of kernel to use.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        padding (bool): Whether to use padding in the convolution.
        stride (int): The stride of the convolution.
        bias (bool): Whether to use bias in the convolution.
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        # Initializing kernel
            
        kernel, _, _, _ = CliffordSteerableKernel(
                algebra=self.algebra,
                c_in=self.c_in,
                c_out=self.c_out,
                kernel_size=self.kernel_size,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                bias_dims=self.bias_dims,
                product_paths_sum=self.product_paths_sum,
                )()

        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (1, self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
            )
            bias = self.algebra.embed(bias_param, self.bias_dims)

        # Reshaping multivector input for compatibiltiy with jax.lax.conv:
        # (N, c_in, X_1, ..., X_dim, 2**algebra.dim) -> (N, c_in * 2**algebra.dim, X_1, ..., X_dim)
        batch_size, input_channels = x.shape[0], self.c_in * self.algebra.n_blades
        spatial_dims = x.shape[-(self.algebra.dim + 1) : -1]
        inputs = (
            jnp.transpose(x, (0, 1, 4, 2, 3))
            if self.algebra.dim == 2
            else jnp.transpose(x, (0, 1, 5, 2, 3, 4))
        )
        inputs = inputs.reshape(batch_size, input_channels, *spatial_dims)

        # Determine padding
        if self.padding_mode.upper() == "SAME":
            padding = "SAME"
        elif self.padding:
            padding = "VALID"
            padding_size = [(self.kernel_size - 1) // 2] * self.algebra.dim
            inputs = jnp.pad(
                inputs,
                [(0, 0), (0, 0)] + [(p, p) for p in padding_size],
                mode=self.padding_mode,
            )
        else:
            padding = "VALID"

        # Convolution
        output = jax.lax.conv(
            inputs,
            kernel,
            window_strides=(self.stride,) * self.algebra.dim,
            padding=padding,
        )

        # Reshaping back to multivector
        output = output.reshape(
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )
        output = (
            jnp.transpose(output, (0, 1, 3, 4, 2))
            if self.algebra.dim == 2
            else jnp.transpose(output, (0, 1, 3, 4, 5, 2))
        )

        if self.bias:
            output = output + bias

        return output


class ComposedCliffordSteerableConv(nn.Module):
    """
    Clifford-steerable convolution layer. See Section 3, Appendix A for details. Pseudocode is given in Function 3.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        kernel_type (string): The type of kernel to use.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        padding (bool): Whether to use padding in the convolution.
        stride (int): The stride of the convolution.
        bias (bool): Whether to use bias in the convolution.
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        # Initializing kernel

        kernel = ComposedCliffordSteerableKernel(
            algebra=self.algebra,
            c_in=self.c_in,
            c_out=self.c_out,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            bias_dims=self.bias_dims,
            product_paths_sum=self.product_paths_sum,
            )()
        
        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (1, self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
            )
            bias = self.algebra.embed(bias_param, self.bias_dims)

        # Reshaping multivector input for compatibiltiy with jax.lax.conv:
        # (N, c_in, X_1, ..., X_dim, 2**algebra.dim) -> (N, c_in * 2**algebra.dim, X_1, ..., X_dim)
        batch_size, input_channels = x.shape[0], self.c_in * self.algebra.n_blades
        spatial_dims = x.shape[-(self.algebra.dim + 1) : -1]
        inputs = (
            jnp.transpose(x, (0, 1, 4, 2, 3))
            if self.algebra.dim == 2
            else jnp.transpose(x, (0, 1, 5, 2, 3, 4))
        )
        inputs = inputs.reshape(batch_size, input_channels, *spatial_dims)

        # Determine padding
        if self.padding_mode.upper() == "SAME":
            padding = "SAME"
        elif self.padding:
            padding = "VALID"
            padding_size = [(self.kernel_size - 1) // 2] * self.algebra.dim
            inputs = jnp.pad(
                inputs,
                [(0, 0), (0, 0)] + [(p, p) for p in padding_size],
                mode=self.padding_mode,
            )
        else:
            padding = "VALID"

        # Convolution
        output = jax.lax.conv(
            inputs,
            kernel,
            window_strides=(self.stride,) * self.algebra.dim,
            padding=padding,
        )

        # Reshaping back to multivector
        output = output.reshape(
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )
        output = (
            jnp.transpose(output, (0, 1, 3, 4, 2))
            if self.algebra.dim == 2
            else jnp.transpose(output, (0, 1, 3, 4, 5, 2))
        )

        if self.bias:
            output = output + bias

        return output
    
def create_circular_mask(kernel_size, center=None, radius=None):
    if center is None:
        center = ((kernel_size-1)/2, (kernel_size-1)/2)
    if radius is None:
        radius = min(center[0], center[1], kernel_size-center[0], kernel_size-center[1])

    Y, X = jnp.ogrid[:kernel_size, :kernel_size]
    dist_from_center = jnp.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(float)

def pool(x, circular_mask):
    return (x * circular_mask[None, :, :, None]).mean((1, 2)) # pool the image into one vector; image has shape [C,X_1,X_2,n_blades] (no batch!)

class ConditionedCliffordSteerableConv(nn.Module):
    """
    Clifford-steerable convolution layer. See Section 3, Appendix A for details. Pseudocode is given in Function 3.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        kernel_type (string): The type of kernel to use.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        padding (bool): Whether to use padding in the convolution.
        stride (int): The stride of the convolution.
        bias (bool): Whether to use bias in the convolution.
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"



    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """

        conv_config = {
            "algebra": self.algebra,
            "c_in": self.c_in,
            "c_out": self.c_out,
            "kernel_size": self.kernel_size,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "bias_dims": self.bias_dims,
            "product_paths_sum": self.product_paths_sum,
            "padding_mode": self.padding_mode,
            "stride": self.stride,
            "bias": self.bias
        }
        
        return jax.vmap(BatchlessConditionedCliffordSteerableConv(**conv_config))(x)



class BatchlessConditionedCliffordSteerableConv(nn.Module):
    """
    Clifford-steerable convolution layer. See Section 3, Appendix A for details. Pseudocode is given in Function 3.

    Attributes:
        algebra (CliffordAlgebra): An instance of CliffordAlgebra defining the algebraic structure.
        c_in (int): The number of input channels.
        c_out (int): The number of output channels.
        kernel_size (int): The size of the kernel.
        kernel_type (string): The type of kernel to use.
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        padding (bool): Whether to use padding in the convolution.
        stride (int): The stride of the convolution.
        bias (bool): Whether to use bias in the convolution.
    """

    algebra: object
    c_in: int
    c_out: int
    kernel_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"
    
    def setup(self):
        """Set up circular mask for pooling"""
        self.circular_mask = create_circular_mask(self.kernel_size)

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        # Initializing kernel

        condition = pool(x, self.circular_mask)

        kernel = CondCliffordSteerableKernel(
                algebra=self.algebra,
                c_in=self.c_in,
                c_out=self.c_out,
                kernel_size=self.kernel_size,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                bias_dims=self.bias_dims,
                product_paths_sum=self.product_paths_sum,
                )(condition)

        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
            )
            bias = self.algebra.embed(bias_param, self.bias_dims)

        # Reshaping multivector input for compatibiltiy with jax.lax.conv:
        # (c_in, X_1, ..., X_dim, 2**algebra.dim) -> (N, c_in * 2**algebra.dim, X_1, ..., X_dim)
        x = x[jnp.newaxis, ...]
        batch_size, input_channels = 1, self.c_in * self.algebra.n_blades
        spatial_dims = x.shape[-(self.algebra.dim + 1) : -1]
        inputs = (
            jnp.transpose(x, (0, 1, 4, 2, 3))
            if self.algebra.dim == 2
            else jnp.transpose(x, (0, 1, 5, 2, 3, 4))
        )
        inputs = inputs.reshape(batch_size, input_channels, *spatial_dims)

        # Determine padding
        if self.padding_mode.upper() == "SAME":
            padding = "SAME"
        elif self.padding:
            padding = "VALID"
            padding_size = [(self.kernel_size - 1) // 2] * self.algebra.dim
            inputs = jnp.pad(
                inputs,
                [(0, 0), (0, 0)] + [(p, p) for p in padding_size],
                mode=self.padding_mode,
            )
        else:
            padding = "VALID"

        # Convolution
        output = jax.lax.conv(
            inputs,
            kernel,
            window_strides=(self.stride,) * self.algebra.dim,
            padding=padding,
        )

        # Reshaping back to multivector
        output = output.reshape(
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )
        output = (
            jnp.transpose(output, (0, 1, 3, 4, 2))
            if self.algebra.dim == 2
            else jnp.transpose(output, (0, 1, 3, 4, 5, 2))
        )

        #remove batch dim
        output = jnp.squeeze(output, axis=0)

        if self.bias:
            output = output + bias

        return output