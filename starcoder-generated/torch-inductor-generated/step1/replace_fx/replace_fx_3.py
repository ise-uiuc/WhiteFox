
m = torch.nn.Identity() + 1
m = nn.Dropout(0.5)(m)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.ones(1, 2, 2)
