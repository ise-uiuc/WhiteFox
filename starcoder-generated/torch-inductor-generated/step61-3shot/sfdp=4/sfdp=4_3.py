
model = Model()
print(model(Q, k, v, mask).shape)
# Inputs to the model
Q = torch.randn(64, 1, 91, 91)
k = torch.randn(64, 1, 60, 60)
v = torch.randn(64, 1, 60, 60)
mask = (torch.rand(64, 91, 91) > 0.7).fill_(-1000000000.0)
