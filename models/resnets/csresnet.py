from flax import linen as nn
import jax.numpy as jnp
import jax

from modules.conv.convolution import CliffordSteerableConv, ComposedCliffordSteerableConv, ConditionedCliffordSteerableConv
from modules.core.norm import MVLayerNorm
from modules.core.mvgelu import MVGELU


class CSBasicBlock(nn.Module):
    """
    Basic block for the Clifford-steerable ResNet.

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        in_channels (int): The number of input channels.
        channels (int): The number of hidden channels.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        norm (bool): Whether to use normalization in the block.
        num_layers (int): The number of layers in the network.
        hidden_dim (int): The number of features in the hidden layers.
        kernel_size (int): The size of the kernel.
        kernel_type (str): The type of kernel to use: "default" or "composed".
        bias_dims (tuple): Dimensions for the bias terms.
        stride (int): The stride of the convolution.
        expansion (int): The expansion factor for the number of channels.
    """

    algebra: object
    in_channels: int
    channels: int
    product_paths_sum: int
    norm: bool
    num_layers: int
    hidden_dim: int
    kernel_size: int
    mask_size: int
    kernel_type: str
    bias_dims: tuple
    stride: int = 1
    expansion: int = 1
    padding_mode: str = "SAME"

    @nn.compact
    def __call__(self, x):
        """
        Applies the basic block to a multivector input.
            x -> conv1 -> norm1 -> gelu -> conv2 -> norm2 -> x + conv(x) -> gelu -> out

        Args:
            x: The input multivector of shape (N, in_channels, X_1, ..., X_dim, 2**algebra.dim).

        Returns:
            The output multivector of shape (N, channels, X_1, ..., X_dim, 2**algebra.dim).
        """
        conv_config = {
            "algebra": self.algebra,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "kernel_size": self.kernel_size,
            "mask_size": self.mask_size,
            "bias_dims": self.bias_dims,
            "product_paths_sum": self.product_paths_sum,
            "padding_mode": self.padding_mode,
        }

        if self.kernel_type == "default":
            Convolution = CliffordSteerableConv
        elif self.kernel_type == "composed":
            Convolution = ComposedCliffordSteerableConv
        elif self.kernel_type == "conditioned":
            Convolution = ConditionedCliffordSteerableConv

        out = Convolution(
            c_in=self.in_channels,
            c_out=self.channels,
            stride=self.stride,
            bias=True,
            **conv_config
        )(x)
        out = MVLayerNorm(self.algebra)(out) if self.norm else out
        out = MVGELU()(out)

        out = Convolution(
            c_in=self.channels,
            c_out=self.channels,
            stride=self.stride,
            padding=True,
            bias=True,
            **conv_config
        )(out)
        out = MVLayerNorm(self.algebra)(out) if self.norm else out

        # shortcut connection
        if self.stride != 1 or self.in_channels != self.expansion * self.channels:
            x = Convolution(
                c_in=self.in_channels,
                c_out=self.expansion * self.channels,
                stride=self.stride,
                padding=True,
                bias=True,
                **conv_config
            )(x)
            x = MVLayerNorm(self.algebra)(x) if self.norm else x

        out += x
        out = MVGELU()(out)
        return out


class CSResNet(nn.Module):
    """
    Clifford-steerable ResNet-based neural solver.
    It takes a stack of fields for multiple time steps and predicts the fields for future time steps.

    Attributes:
        algebra (object): An instance of CliffordAlgebra defining the algebraic structure.
        time_history (int): The number of input channels.
        time_future (int): The number of output channels.
        hidden_channels (int): The number of hidden channels.
        kernel_num_layers (int): The number of layers in the network.
        kernel_hidden_dim (int): The number of features in the hidden layers.
        kernel_size (int): The size of the kernel.
        kernel_type (str): The type of kernel to use: "default" or "composed".
        bias_dims (tuple): Dimensions for the bias terms.
        product_paths_sum (int): The number of non-zero elements in the Cayley table.
            - given by algebra.geometric_product_paths.sum().item()
        blocks (tuple): The number of blocks in the network.
        norm (bool): Whether to use normalization in the network.
        make_channels (bool): Whether to use the input and output channels as features.
            - only used for non-Euclidean data.
    """

    algebra: object
    time_history: int
    time_future: int
    hidden_channels: int
    kernel_num_layers: int
    kernel_hidden_dim: int
    kernel_size: int
    kernel_type: str
    bias_dims: tuple
    product_paths_sum: int
    blocks: tuple = (2, 2, 2, 2)
    norm: bool = True
    make_channels: bool = False
    padding_mode: str = "SAME"

    def setup(self):
        self.conv_config = {
            "algebra": self.algebra,
            "kernel_size": 1,
            "bias_dims": self.bias_dims,
            "num_layers": self.kernel_num_layers,
            "hidden_dim": self.kernel_hidden_dim,
            "product_paths_sum": self.product_paths_sum,
            "padding_mode": self.padding_mode,
        }

        self.block_config = {
            "algebra": self.algebra,
            "in_channels": self.hidden_channels,
            "channels": self.hidden_channels,
            "product_paths_sum": self.product_paths_sum,
            "norm": self.norm,
            "num_layers": self.kernel_num_layers,
            "hidden_dim": self.kernel_hidden_dim,
            "kernel_size": self.kernel_size,
            "kernel_type": self.kernel_type,
            "bias_dims": self.bias_dims,
            "padding_mode": self.padding_mode,
        }

    @nn.compact
    def __call__(self, x):
        """
        Forward pass of the model.
            x -> 2 1x1 convolutions -> N basic blocks -> 2 1x1 convolutions -> out

        Args:
            x: The input multivector of shape (N, time_history, X_1, ..., X_dim, 2**algebra.dim).

        Returns:
            The output multivector of shape (N, time_future, X_1, ..., X_dim, 2**algebra.dim).
        """
        # Embedding convolutional layers
        in_channels = self.time_history if not self.make_channels else 1
        out_channels = self.time_future if not self.make_channels else 1
        if self.algebra.dim == 2:
            mask_size = 64
        elif self.algebra.dim == 3:
            mask_size = 32

        if self.kernel_type == "default":
            Convolution = CliffordSteerableConv
        elif self.kernel_type == "composed":
            Convolution = ComposedCliffordSteerableConv
        elif self.kernel_type == "conditioned":
            Convolution = ConditionedCliffordSteerableConv

        x = Convolution(
            c_in=in_channels, c_out=self.hidden_channels, mask_size=mask_size,  **self.conv_config
        )(x)
        x = MVGELU()(x)
        x = Convolution(
            c_in=self.hidden_channels, c_out=self.hidden_channels, mask_size=mask_size, **self.conv_config
        )(x)
        x = MVGELU()(x)

        # Basic blocks
        for num_blocks in self.blocks:
            for _ in range(num_blocks):
                x = CSBasicBlock(mask_size=mask_size, **self.block_config)(x)

        # Output convolutional layers
        x = Convolution(
            c_in=self.hidden_channels, c_out=self.hidden_channels, mask_size=mask_size,  **self.conv_config
        )(x)
        x = MVGELU()(x)
        x = Convolution(
            c_in=self.hidden_channels, c_out=out_channels, mask_size=mask_size, **self.conv_config
        )(x)
        return x
