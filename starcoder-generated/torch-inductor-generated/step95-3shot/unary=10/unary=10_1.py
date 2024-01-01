
m = torch.nn.Sequential(torch.nn.Linear(12, 7), torch.nn.ReLU6(True), torch.nn.Linear(7, 5))

# Inputs to the model
x = torch.randn(20, 12)
