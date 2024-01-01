
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(64, 3, 1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(3, 1, 1),
    torch.nn.ReLU()
)

x2 = torch.randn(1, 3, 64, 64)
