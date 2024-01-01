
self.linear = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.Sigmoid(), torch.nn.Linear(128, 195), torch.nn.ReLU())

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 64)
