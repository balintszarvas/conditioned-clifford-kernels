#
#
#THIS IS A TRIAL FOR HANDLING BATCHES IN THE CONDITIONED CONVOLUTION, NOT IMPLEMENTABLE YET
#
#


import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros

from .kernel import CliffordSteerableKernel
from .composedkernel import ComposedCliffordSteerableKernel
from .condkernel import CondCliffordSteerableKernel

def create_circular_mask(kernel_size, center=None, radius=None):
    if center is None:  # use the middle of the image
        center = ((kernel_size-1)/2, (kernel_size-1)/2)
    if radius is None:
        radius = min(center[0], center[1], kernel_size-center[0], kernel_size-center[1])

    Y, X = jnp.ogrid[:kernel_size, :kernel_size]
    dist_from_center = jnp.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(float)

def pool(x, circular_mask, algebra):
    """
    Pools the input tensor x over spatial dimensions using the provided circular_mask.

    Args:
        x: Input tensor of shape [batch_size, c_in, *spatial_dims, n_blades]
        circular_mask: Mask tensor of shape [kernel_size, ...] with spatial dimensions matching kernel size

    Returns:
        Pooled tensor of shape [batch_size, c_in, n_blades]
    """
    # Expand circular_mask to match spatial dimensions
    mask = circular_mask[None, None, *([Ellipsis] + [None] * (x.ndim - 3)), None]  # Adds batch and channel dims

    # Apply mask and pool over spatial dimensions
    x_masked = x * mask
    pooled = x_masked.mean(axis=tuple(range(2, 2 + algebra.dim)))  # Mean over spatial dimensions

    return pooled


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
    kernel_type: str # "default" , "conditioned" or "composed"
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

        if self.kernel_type == "composed":
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
            
        elif self.kernel_type == "conditioned":
            
            condition = pool(x, self.circular_mask)

            kernel = CondCliffordSteerableKernel( # this doesn't take batches
                algebra=self.algebra,
                c_in=self.c_in,
                c_out=self.c_out,
                kernel_size=self.kernel_size,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                bias_dims=self.bias_dims,
                product_paths_sum=self.product_paths_sum,
                )(condition)
        else:
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
        
        # Calculate the total number of kernel elements
        kernel_elements = self.c_in * self.algebra.n_blades * (self.kernel_size ** self.algebra.dim)

        # Reshape the kernel accordingly
        kernel = kernel.reshape(
            batch_size,
            self.c_out * self.algebra.n_blades,
            kernel_elements
        )
        
        kernel = kernel.transpose(0, 2, 1)  # Shape: (batch_size, kernel_elements, c_out * n_blades)

        # Extract patches
        window_shape = (self.kernel_size,) * self.algebra.dim
        window_strides = (self.stride,) * self.algebra.dim

        patches = jax.lax.conv_general_dilated_patches( # shape: (batch_size, num_patches, c_in * n_blades * kernel_size**dim)
            inputs,
            window_shape=window_shape,
            window_strides=window_strides,
            padding=padding,
        )

        # Convolution
        output = jnp.einsum('bpi, bio -> bpo', patches, kernel) # shape: (batch_size, num_patches, c_out * n_blades)

        # Compute output spatial dimensions
        input_spatial_dims = inputs.shape[2:]
        if padding == 'SAME':
            output_spatial_dims = [dim_size // self.stride for dim_size in input_spatial_dims]
        else:
            output_spatial_dims = [
                (dim_size - self.kernel_size) // self.stride + 1 for dim_size in input_spatial_dims
            ]

        # Reshape output to spatial dimensions
        output = output.reshape(
            batch_size,
            *output_spatial_dims,
            self.c_out,
            self.algebra.n_blades
        )

        if self.bias:
            output = output + bias

        return output