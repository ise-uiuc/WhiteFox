
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v1 = v1.transpose(0, 1)
        return torch.cat([v1], dim=0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 3)
