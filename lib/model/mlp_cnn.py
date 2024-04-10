from typing import Callable, List, Optional, Union
import torch
from torch import nn
from torch.types import _dtype


class MLP(nn.Module):
    """A multi-layer perceptron module.

    This module is a sequence of linear layers plus activation functions.
    The user can optionally add normalization and/or dropout to each of the layers.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
        dtype: Optional[_dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            if normalization:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=False,
                                        dtype=dtype))
                layers.append(normalization(hidden_dim, dtype=dtype))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim, dtype=dtype))
            layers.append(activation())
            if dropout != 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MLP_Conv2d(nn.Module):
    """A multi-layer perceptron module implemented as 1x1 convolutions.

    This module is a sequence of 1x1 2D convolutions layers plus activation
    functions.
    Optionally normalization and/or dropout to each of the layers may be added.

    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Optional[List[int]]): Output dimension for each hidden layer.
        dropout (float): Probability for dropout layer.
        activation (Callable[..., nn.Module]): Which activation
            function to use. Supports module type or partial.
        normalization (Optional[Callable[..., nn.Module]]): Which
            normalization layer to use (None for no normalization).
            Supports module type or partial.

    Inputs:
        x (Tensor): Tensor containing a batch of input sequences.
    ​
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[Union[int, List[int]]] = None,
        dropout: float = 0.0,
        activation: Callable[..., nn.Module] = nn.ReLU,
        normalization: Optional[Callable[..., nn.Module]] = None,
        dtype: Optional[_dtype] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        layers = nn.ModuleList()

        if hidden_dims is None:
            hidden_dims = []

        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        for hidden_dim in hidden_dims:
            
            if normalization:
                # If normalization layer is used then 'bias' is set to False
                layers.append(nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim,
                                        kernel_size=1, stride=1, padding=0,
                                        dtype=dtype, bias=False))
                layers.append(normalization(hidden_dim, dtype=dtype))
            else:
                layers.append(nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim,
                                        kernel_size=1, stride=1, padding=0,
                                        dtype=dtype))
            layers.append(activation())
            if dropout != 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Conv2d(in_channels=in_dim, out_channels=out_dim,
                                kernel_size=1, stride=1, padding=0,
                                dtype=dtype))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)