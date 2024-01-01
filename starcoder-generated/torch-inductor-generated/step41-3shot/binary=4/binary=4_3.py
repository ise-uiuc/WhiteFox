
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1, x2, x3):
        x4 = torch.cat([x1, x2, x3], dim=0)
        v1 = self.linear(x4)
        v2 = v1 + x1
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
x2 = torch.randn(1, 8)
x3 = torch.randn(1, 8)
