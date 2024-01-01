
model = Model()
y = model(x1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
y = model(x1)
