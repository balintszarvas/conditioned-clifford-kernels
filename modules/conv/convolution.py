import jax
import time
import numpy as np
from jax import profiler
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import zeros
from functools import partial

from .kernel import CliffordSteerableKernel
from .composedkernel import ComposedCliffordSteerableKernel
from .condkernel import CondCliffordSteerableKernel, BatchedCondCliffordSteerableKernel
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
    batch_size: int = 1

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

def pool_2d(x, circular_mask, algebra):
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
    # x shape: [1,X_1,X_2,X_3,n_blades]
    # circular_mask shape: [mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, 
                               circular_mask.shape[0], 
                               circular_mask.shape[1], 
                               circular_mask.shape[2], 
                               1)
    
    # Apply mask and average over spatial dimensions (1,2,3)
    x =(x * mask).mean(axis=(1, 2, 3))
    jax.debug.print("Condition shape after spatial pooling: {shape}", shape=x.shape)

    x = x.mean(axis=0)
    x = x[jnp.newaxis, ...]
    jax.debug.print("Condition shape after channel pooling: {shape}", shape=x.shape)
    x = MVLayerNorm(algebra=algebra)(x)
    jax.debug.print("Condition shape after norm: {shape}", shape=x.shape)
    return x



# TODO: I WAS TRYING TO IMPLEMENT A NEW POOLING FUNCTION THAT WOULD JIT COMPILE BETTER IN 3D, BUT ITS NOT REALLY THE BOTTLENECK.
def new_pooling(x, circular_mask):
    """"Applies pooling to a multivector input"""
    # x shape: (B, C, X, Y, Z, n_blades)
    x = x.transpose(0,5,1,2,3,4)               # -> (B, n_blades, C, X, Y, Z)
    b, bld, c, *spatial_dims = x.shape

    # Collapse channel axis into the feature dimension expected by conv
    # Input becomes (B, bld*C, X, Y, Z) and we will use feature_group_count=bld
    x_flat = x.reshape(b, bld*c, *spatial_dims)

    # ------------------------------------------------------------------
    # Build a kernel that (i) averages over spatial positions via the
    #     circular mask, and (ii) averages over the *channel* dimension.
    # Using groups = n_blades means each blade group reduces its own C
    # channels to a single output, giving us exactly one scalar per blade.
    # ------------------------------------------------------------------

    # Normalise the spatial mask so that it sums to 1
    mask_kernel = (circular_mask / circular_mask.sum()).astype(x.dtype)

    # Kernel shape required by conv: (C_out, C_in_per_group, k, k, k)
    # We want C_out = bld, groups = bld, so C_in_per_group = C
    kernel = jnp.broadcast_to(
        mask_kernel, (bld, c) + mask_kernel.shape
    ) / c  # divide by channels to perform channel averaging

    cond = jax.lax.conv_general_dilated(
        x_flat,
        kernel,
        window_strides=(1,)*len(spatial_dims),
        padding="VALID",
        dimension_numbers=("NCDHW","OIDHW","NCDHW"),
        feature_group_count=bld,
    )

    # cond shape now (B, bld, 1, 1, 1); drop singleton spatial dims
    cond = cond.squeeze(tuple(range(2, 2+len(spatial_dims))))  # -> (B, bld)

    # Insert singleton channel axis so the overall shape is (B, 1, n_blades)
    return cond[:, jnp.newaxis, :]

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
    batch_size: int = 1
    
    def setup(self):
        """
        Selects the pooling function based on the algebra dimension
        """
        metric = np.asarray(self.algebra.metric)
        if self.algebra.dim == 2:
            #print("Using 2D non-relativistic pooling")
            self.pool_fn = pool_2d
        elif metric.sum() == 3 and self.algebra.dim == 3:
            #print("Using 3D non-relativistic pooling")
            self.pool_fn = pool_3d
        else:
            #print("Using 2D relativistic pooling")
            self.pool_fn = pool_2d_relativistic
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
        condition = jax.vmap(self.pool_fn, in_axes=(0, None, None))(x, self.circular_mask, self.algebra)
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
    batch_size: int = 1


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

        self.dummy_condition = jnp.ones([self.c_in, 2**self.algebra.dim], dtype=jnp.float32)

        print("Dummy condition shape", self.dummy_condition.shape)
        
        # Define a single-sample kernel generator (shares parameters)
        self.kernel_single = BatchedCondCliffordSteerableKernel(
            algebra=self.algebra,
            c_in=self.c_in,
            c_out=self.c_out,
            kernel_size=self.kernel_size,
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            bias_dims=self.bias_dims,
            product_paths_sum=self.product_paths_sum,
        )

        vars_ = self.kernel_single.init(jax.random.PRNGKey(42), self.dummy_condition)

        # Batched generator: vmap over the leading axis (condition batch).
        # Because kernel_single is captured as a Flax sub-module, it
        # automatically reuses the same parameters; no explicit variable
        # tree needs to be passed.
        per_sample = partial(self.kernel_single.apply, vars_)
        self.kernel_batched = jax.jit(
            jax.vmap(per_sample, in_axes=0)
        )

    @nn.compact
    def __call__(self, x):
        """
        Applies the convolution operation to a multivector input.
        Args:
        x: The input multivector of shape (B, c_in, X_1, ..., X_dim, 2**algebra.dim).
        Returns:
        The output multivector of shape (B, c_out, X_1, ..., X_dim, 2**algebra.dim).
        """
        print("x shape on input to condition", x.shape)

        condition = self.pool_fn(x, self.circular_mask)

        print("Condition shape", condition.shape)

        kernels = self.kernel_batched(condition)
        
        # Initializing bias
        if self.bias:
            bias_param = self.param(
                "bias",
                zeros,
                (self.c_out, *([1] * self.algebra.dim), len(self.bias_dims)),
            )
            bias = self.algebra.embed(bias_param, self.bias_dims)

        # Reshaping multivector input for compatibiltiy with jax.lax.conv:
        # (B, c_in, X_1, ..., X_dim, 2**algebra.dim) -> (B, c_in * 2**algebra.dim, X_1, ..., X_dim)
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
        CoutBlades, CinBlades = kernels.shape[1], kernels.shape[2]
        kernels = kernels.reshape(batch_size*CoutBlades, CinBlades, *[self.kernel_size]*self.algebra.dim)

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

        print("Inputs shape", inputs.shape, "Kernels shape", kernels.shape)

        # Convolution
        output = jax.lax.conv_general_dilated(
            inputs,                    # (1, B·Cin * n_blades, X, Y, Z)
            kernels,                   # (B·Cout * n_blades, Cin * n_blades, k, k, k)
            window_strides=(self.stride,)*self.algebra.dim,
            padding=padding,
            dimension_numbers=("NCDHW","OIDHW","NCDHW"),
            feature_group_count=batch_size,
            )
        
        print("Output shape directly after conv", output.shape)

        # Reshaping back to multivector
        output = output.reshape(
            batch_size,
            self.c_out,
            self.algebra.n_blades,
            *output.shape[-self.algebra.dim :]
        )

        print("Output shape", output.shape)
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

def batched_pool_2d(x, circular_mask, algebra):
    """2D pooling with fixed dimensions"""
    # x shape: [C,X_1,X_2,n_blades]
    # circular_mask shape: [mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, circular_mask.shape[0], circular_mask.shape[1], 1)
    
    # Apply mask and average over spatial dimensions (1,2)
    return (x * mask).mean(axis=(1, 2))

def batched_pool_3d(x, circular_mask, algebra):
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
def batched_pool_2d_relativistic(x, circular_mask, algebra):
    """2D pooling with fixed dimensions"""
    # x shape: [1,X_1,X_2,X_3,n_blades]
    # circular_mask shape: [mask_size, mask_size]
    
    # Reshape mask with explicit dimensions
    mask = circular_mask.reshape(1, 
                               circular_mask.shape[0], 
                               circular_mask.shape[1], 
                               circular_mask.shape[2], 
                               1)
    
    # Apply mask and average over spatial dimensions (1,2,3)
    x =(x * mask).mean(axis=(1, 2, 3))
    #jax.debug.print("Condition shape after spatial pooling: {shape}", shape=x.shape)

    x = x.mean(axis=0)
    x = x[jnp.newaxis, ...]
    #jax.debug.print("Condition shape after channel pooling: {shape}", shape=x.shape)
    x = MVLayerNorm(algebra=algebra)(x)
    #jax.debug.print("Condition shape after norm: {shape}", shape=x.shape)
    return x



class AltConditionedCliffordSteerableConv(nn.Module):
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
    batch_size: int = 1

    def setup(self):
        """
        Selects the pooling function based on the algebra dimension
        """
        metric = np.asarray(self.algebra.metric)
        if self.algebra.dim == 2:
            #print("Using 2D non-relativistic pooling")
            self.pool_fn = pool_2d
        elif metric.sum() == 3 and self.algebra.dim == 3:
            #print("Using 3D non-relativistic pooling")
            self.pool_fn = pool_3d
        else:
            #print("Using 2D relativistic pooling")
            self.pool_fn = pool_2d_relativistic
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
        # Initializing kernel
        avgd_x = jnp.mean(x, axis=0)
        condition = self.pool_fn(avgd_x, self.circular_mask, self.algebra)

        # #returns rel_pos, factor and weighted_cayley additionally for the composed kernel, not necessary if we dont include it in the final code
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