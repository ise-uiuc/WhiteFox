
model = torch.nn.Linear(4, 3)
model.weight.data.fill_(1)
model.bias.data.fill_(1)
torch.onnx.export(model, x, "model.onnx", verbose=False)
# Inputs to the model
x = (torch.rand(3, 2), torch.rand(3, 2), torch.rand(3, 2))
