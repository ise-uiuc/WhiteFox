
model = SimpleModel()
model.addSubmodule("conv",torch.nn.Conv2d(3, 16, 3, stride=1, padding=1))
model.addSubmodule("tanh",torch.nn.Tanh())
# Add multiple submodules to the model
# Inputs to the model
x3 = torch.randn(64, 3, 64, 64)
