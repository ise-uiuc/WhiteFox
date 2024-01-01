
nn = nn.Sequential(
    torch.nn.Linear(50, 12),
    torch.nn.ReLU(inplace=True),
    torch.nn.Linear(12, 4),
)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 50, requires_grad=True)
x2 = torch.randn(1, 50, requires_grad=True)
x3 = torch.randn(1, 50, requires_grad=True)
x4 = torch.randn(1, 50, requires_grad=True)
x5 = torch.randn(1, 50, requires_grad=True)
inputs = (x1, x2, x3, x4, x5)
