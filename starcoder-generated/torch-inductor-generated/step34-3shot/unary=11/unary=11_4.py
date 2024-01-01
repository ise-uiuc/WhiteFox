
model = torch.nn.ConvTranspose2d(3, 32, 5, stride=1, padding=2)
model.weight = torch.randn(32, 3, 5, 5)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
