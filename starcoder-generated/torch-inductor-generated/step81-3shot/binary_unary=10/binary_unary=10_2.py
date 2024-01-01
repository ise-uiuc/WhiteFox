
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(6, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        return torch.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 6)
