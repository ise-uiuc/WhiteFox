
__model__ = torch.nn.Sequential(torch.nn.Linear(10, 1))

# Initializing the model
m = __model__

# Inputs to the model
x1 = torch.randn(1, 10)
