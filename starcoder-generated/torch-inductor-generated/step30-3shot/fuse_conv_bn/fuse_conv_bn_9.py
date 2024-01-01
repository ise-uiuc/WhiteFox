
model = torch.nn.Sequential(
    torch.nn.Conv2d(2, 32, 5, 1, 2),
    torch.nn.BatchNorm2d(32),
    torch.nn.Conv2d(32, 64, 5, 1, 2),
    torch.nn.ReLU(),
    torch.nn.Linear(9216, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 10)
)
# Inputs to the model
x = torch.randn(2, 2, 28, 28)
