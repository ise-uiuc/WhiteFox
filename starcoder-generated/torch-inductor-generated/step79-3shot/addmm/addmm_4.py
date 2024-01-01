
model = torch.nn.Sequential(
        torch.nn.Linear(100, 100),
        torch.nn.Linear(100, 100))
# Inputs to the model
inp = torch.randn(3, 3, requires_grad=True)
