
model = torch.nn.Sequential(torch.nn.Conv2d(7, 4, 2, stride=1, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(4, 7, 2, stride=1, padding=1))
x1 = torch.randn(1, 7, 64, 64)
