
conv = torch.nn.Conv2d(3, 8, 13, stride=21, padding=25)
model = Model(conv=conv)

x1 = torch.randn(1, 3, 42, 42)
