
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64+32, 32)
 
    def forward(self, x1, x2):
        v1 = self.linear(torch.cat([x1, x2], dim=1))
        v2 = torch.nn.functional.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
x2 = torch.randn(1, 32)
