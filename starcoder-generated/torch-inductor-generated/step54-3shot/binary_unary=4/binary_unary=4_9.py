
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1 + 1, 6)
 
    def forward(self, x1, x2):
        v1 = self.linear(torch.cat([x1, x2]))
        v2 = v1 + x2
        v3 = F.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
