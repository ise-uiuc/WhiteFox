
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x1):
        a = torch.relu(self.linear(x1))
        b = a - torch.tensor([[1, 2]])
        return self.linear(b)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
