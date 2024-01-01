
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 1, kernel_size=1),
    torch.nn.Conv2d(1, 3, kernel_size=1)
)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
