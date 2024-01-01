
model = torch.nn.Sequential(
    torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
)

X1 = torch.rand(1, 8, 64, 64)
