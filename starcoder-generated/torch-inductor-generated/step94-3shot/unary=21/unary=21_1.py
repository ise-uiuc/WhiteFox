
model = torch.nn.Conv2d(3, 1, [1, 20], padding=[0, 0])
# Inputs to the model
x = torch.randn(1, 3, 65, 65)
