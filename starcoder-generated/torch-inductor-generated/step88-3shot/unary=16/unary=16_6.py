
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(5, 5),
            torch.nn.Linear(5, 1)
        )
 
    def forward(self, x1):
        v1 = self.linears(x1)
        v2 = torch.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(4, 5)
