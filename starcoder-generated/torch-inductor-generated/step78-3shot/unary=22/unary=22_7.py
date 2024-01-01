
model = torch.nn.Sequential(torch.nn.Linear(3, 8),
    torch.nn.Tanh())

# Inputs to the model
x1 = torch.randn(1, 3)
