
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + torch.ones(1, 8, 1, 1)
        v3 = torch.relu(v2)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
