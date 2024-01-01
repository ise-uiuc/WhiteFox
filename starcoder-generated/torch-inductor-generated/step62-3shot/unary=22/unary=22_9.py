
model = Torch.torch.nn.Sequential(
    Torch.torch.nn.Conv2d(
        in_channels: 3,
        out_channels: 64,
        kernel_size: (1, 1),
        stride: (1, 1)
    ),
    Torch.torch.nn.Flatten(start_dim=1, end_dim=-1),
    Torch.torch.nn.Linear(
        in_features: 64*64,
        out_features: 3072
    ),
    Torch.torch.nn.Tanh(),
)

# Initializing the model
m = model

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
