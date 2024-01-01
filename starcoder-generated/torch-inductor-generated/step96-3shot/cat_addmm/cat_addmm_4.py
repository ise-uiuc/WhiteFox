
conv = nn.Conv2d(1, 1, 1)
conv.weight = nn.Parameter(torch.randn(1, 1, 1, 1))
x = torch.randn(1, 1, 1, 1)
with torch.no_grad():
  conv(x)
