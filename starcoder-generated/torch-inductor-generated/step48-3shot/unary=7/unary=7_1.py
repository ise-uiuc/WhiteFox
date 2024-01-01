
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.clamp(v1, min=0, max=6) + 3
        v3 = v2 / 6
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
y1 = m(x1)

x2 = torch.randn(10, 8)
y2 = m(x2)

x3 = torch.randn(16, 8)
y3 = m(x3)

