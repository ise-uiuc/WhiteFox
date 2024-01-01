
model = torch.jit.script(Model(min_value=-1., max_value=5.))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
