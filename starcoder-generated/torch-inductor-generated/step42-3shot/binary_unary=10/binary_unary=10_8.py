
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x2, x3):
        v1 = self.linear(x2 + x3)
        v2 = F.relu(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 8)
