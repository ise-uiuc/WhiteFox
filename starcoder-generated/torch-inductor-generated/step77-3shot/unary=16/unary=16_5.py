
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        v1 = torch.matmul(x, torch.tensor([[.7, -.2, 0.5]]))
        return torch.relu(v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 2)

