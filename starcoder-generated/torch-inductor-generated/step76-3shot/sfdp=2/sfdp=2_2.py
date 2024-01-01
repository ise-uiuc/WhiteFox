 output shapes
m = AttentionModel()
x1 = torch.randn(1, 1, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
x3 = torch.randn(1, 1, 64, 64)
__output__, weights, scores = m(x1, x2, x3)

