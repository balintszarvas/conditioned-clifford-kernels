import jax
import time
from jax import profiler
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros
import functools

from .kernel import CliffordSteerableKernel
from .composedkernel import ComposedCliffordSteerableKernel
from .condkernel import CondCliffordSteerableKernel
from modules.core.norm import MVLayerNorm



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
    mask_size: int
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
        # #returns rel_pos, factor and weighted_cayley additionally for the composed kernel, not necessary if we dont include it in the final code
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
    
def create_circular_mask(kernel_size, center=None, radius=None, dim=2):
    """Create circular/spherical mask for 2D/3D"""
    if center is None:
        center = tuple((kernel_size-1)/2 for _ in range(dim))
    if radius is None:
        radius = min(center[0], center[1], kernel_size-center[0], kernel_size-center[1])
    if dim == 2:
        Y, X = jnp.ogrid[:kernel_size, :kernel_size]
        dist_from_center = jnp.sqrt((X - center[0])**2 + (Y-center[1])**2)
    elif dim == 3:
        Z, Y, X = jnp.ogrid[:kernel_size, :kernel_size, :kernel_size]
        dist_from_center = jnp.sqrt((X - center[0])**2 + (Y-center[1])**2 + (Z-center[2])**2)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
    
    mask = dist_from_center <= radius
    return mask.astype(float)

def pool_2d(x, circular_mask):
    """2D pooling with fixed dimensions"""
    # x shape: [C,X_1,X_2,n_blades]
    # circular_mask shape: [mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, circular_mask.shape[0], circular_mask.shape[1], 1)
    
    # Apply mask and average over spatial dimensions (1,2)
    return (x * mask).mean(axis=(1, 2))

def pool_3d(x, circular_mask, algebra):
    """3D pooling with fixed dimensions"""
    # x shape: [C,X_1,X_2,X_3,n_blades]
    # circular_mask shape: [mask_size, mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, 
                               circular_mask.shape[0], 
                               circular_mask.shape[1], 
                               circular_mask.shape[2], 
                               1)
    
    # Apply mask and average over spatial dimensions (1,2,3)
    return (x * mask).mean(axis=(1, 2, 3))

# TODO: THIS IS AGAIN ME TRYING TO NORMALISE THE CONDITION SO THAT THE VALUES DON'T EXPLODE TO NAN IN THE 2D MAXWELL SIM
def pool_2d_relativistic(x, circular_mask, algebra):
    """2D pooling with fixed dimensions"""
    # x shape: [C,X_1,X_2,X_3,n_blades]
    # circular_mask shape: [mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, 
                               circular_mask.shape[0], 
                               circular_mask.shape[1], 
                               circular_mask.shape[2], 
                               1)
    
    # Apply mask and average over spatial dimensions (1,2,3)
    x =(x * mask).mean(axis=(1, 2, 3))
    return MVLayerNorm(algebra=algebra)(x)

# TODO: I WAS TRYING TO IMPLEMENT A NEW POOLING FUNCTION THAT WOULD JIT COMPILE BETTER IN 3D, BUT ITS NOT REALLY THE BOTTLENECK.
def new_pooling(x, circular_mask):
    """"Applies pooling to a multivector input"""
    x = x.transpose(0,1,5,2,3,4)
    b, c, bld, *spatial_dims = x.shape

    x_flat = x.reshape(b, c*bld, *spatial_dims)
    mask_kernel = (circular_mask / circular_mask.sum()).astype(x.dtype)

    rhs = jnp.broadcast_to(mask_kernel[None, None, ...], (c*bld, 1, *mask_kernel.shape))

    cond_flat = jax.lax.conv_general_dilated(
        x_flat,
        rhs,
        window_strides=(1,1,1),
        padding="VALID",
        dimension_numbers=("NCDHW","OIDHW","NCDHW"),
        feature_group_count=c*bld,
    )

    return cond_flat.squeeze((2,3,4)).reshape(b, c, bld)

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
    mask_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"
    
    def setup(self):
        """
        Selects the pooling function based on the algebra dimension
        """
        self.pool_fn = pool_2d if self.algebra.dim == 2 else pool_3d
        #self.pool_fn = new_pooling
        #self.circular_mask = jnp.ones([self.mask_size, self.mask_size, self.mask_size])
        self.circular_mask = create_circular_mask(
            self.mask_size, dim=self.algebra.dim
        )

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (N, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (N, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        #bcondconv_start = time.perf_counter()
        conv_config = {
            "algebra": self.algebra,
            "c_in": self.c_in,
            "c_out": self.c_out,
            "kernel_size": self.kernel_size,
            "mask_size": self.mask_size,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "bias_dims": self.bias_dims,
            "product_paths_sum": self.product_paths_sum,
            "padding_mode": self.padding_mode,
            "stride": self.stride,
            "bias": self.bias
        }
        #condition_start = time.perf_counter()
        condition = jax.vmap(self.pool_fn, in_axes=(0, None))(x, self.circular_mask)
        #condition = jnp.ones([2, self.c_in, 2**self.algebra.dim])
        #condition = self.pool_fn(x, self.circular_mask)
        #condition_end = time.perf_counter()
        #print("Condition compute time", (condition_end - condition_start))
        output = jax.vmap(BatchlessConditionedCliffordSteerableConv(**conv_config))(x, condition)
        #bcondconv_end = time.perf_counter()
        #print("Batched condconv time:", (bcondconv_end - bcondconv_start))
        return output
    
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
    mask_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"

    @nn.compact
    def __call__(self, x, condition):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (c_out, X_1, ..., X_dim, 2**algebra.dim).
        """

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

# ANOTHER IMPLEMENTATION OF THE CONDITIONED CONV. AIMING TO JIT COMPLIE BETTER IN 3D, BUT NOT WORKING.
class NewConditionedCliffordSteerableConv(nn.Module):
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
    mask_size: int
    bias_dims: tuple
    product_paths_sum: int
    num_layers: int
    hidden_dim: int
    padding: bool = True
    stride: int = 1
    bias: bool = True
    padding_mode: str = "SAME"


    def setup(self):
        """
        Selects the pooling function based on the algebra dimension
        """
        #self.pool_fn = pool_2d if self.algebra.dim == 2 else pool_3d
        self.pool_fn = new_pooling
        #self.circular_mask = jnp.ones([self.mask_size, self.mask_size, self.mask_size])
        self.circular_mask = create_circular_mask(
            self.mask_size, dim=self.algebra.dim
        )

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (c_out, X_1, ..., X_dim, 2**algebra.dim).
        """

        condition = self.pool_fn(x, self.circular_mask)

        kernels = jax.vmap(
            CondCliffordSteerableKernel(
                algebra=self.algebra,
                c_in=self.c_in,
                c_out=self.c_out,
                kernel_size=self.kernel_size,
                num_layers=self.num_layers,
                hidden_dim=self.hidden_dim,
                bias_dims=self.bias_dims,
                product_paths_sum=self.product_paths_sum,
                ))(condition)
        
        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
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
        inputs = inputs[jnp.newaxis, ...].reshape(1, batch_size*input_channels, *spatial_dims)

        # Reshapekernels: (N, c_out * 2**algebra.dim, c_in * 2**algebra.dim, X_1, ..., X_dim) -> 
        # -> (N*c_out * 2**algebra.dim, c_in * 2**algebra.dim, X_1, ..., X_dim)
        CoutB, CinB = kernels.shape[1], kernels.shape[2]
        kernels = kernels.reshape(batch_size*CoutB, CinB, *[self.kernel_size]*self.algebra.dim)

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
        output = jax.lax.conv_general_dilated(
            inputs,                    # (1, B·CinB, X, Y, Z)
            kernels,                   # (B·CoutB, CinB, k, k, k)
            window_strides=(self.stride,)*self.algebra.dim,
            padding=padding,
            dimension_numbers=("NCDHW","OIDHW","NCDHW"),
            feature_group_count=batch_size,
            )  

        # Reshaping back to multivector
        output = output.reshape(
            1,
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )[0]
        output = (
            jnp.transpose(output, (0, 1, 3, 4, 2))
            if self.algebra.dim == 2
            else jnp.transpose(output, (0, 1, 3, 4, 5, 2))
        )


        if self.bias:
            output = output + bias
        
        return output

########################################################################################
# Composed Clifford Steerable Conv (not in paper)
########################################################################################

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
    mask_size: int
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