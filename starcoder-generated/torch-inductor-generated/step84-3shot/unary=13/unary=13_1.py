
Model(
  (linear1): torch.nn.Linear(16, 16)
  (linear2): torch.nn.Linear(16, 1)
  (sigmoid): torch.nn.Sigmoid()
)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(10, 16)
