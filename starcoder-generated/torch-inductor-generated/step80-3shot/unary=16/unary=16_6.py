
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1).t()
        return torch.nn.functional.relu(v1)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(128, 64)
