
model = torch.nn.Sequential(
    torch.nn.Hardtanh(),
    torch.nn.AdaptiveAvgPool2d(),
    torch.nn.Conv1d(),
    torch.nn.Sequential(
        torch.nn.ConvTranspose1d(),
        torch.nn.Dropout(),
        torch.nn.BatchNorm1d(requires_grad=True),
    ),
    torch.nn.Hardtanh(),
)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
