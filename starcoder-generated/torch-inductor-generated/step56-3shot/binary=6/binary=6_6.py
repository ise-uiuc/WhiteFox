
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=True)
 
    def forward(self, x1, x2):
        x1 = x1 + 1
        v1 = self.linear(x1)
        v2 = v1 - x2
        return v2

# Initializing the model
m1 = Model()

# Inputs to the model
x1 = torch.randn(2, 64)
x2 = torch.randn(2, 64)
