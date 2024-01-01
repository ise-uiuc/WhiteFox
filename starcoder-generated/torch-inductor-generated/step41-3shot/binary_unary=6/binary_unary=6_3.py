
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        __v3__ = torch.tensor([[1.0, 2.0, 3.0]])
        v3 = torch.randn_like(__v3__)
        v2 = v1 - v3
        v4 = torch.relu(v2)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
