
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.clamp(min=0, max=6)
        v3 = v2 + 3
        return v3 / 6

# Initializing the model
m = Model()

# Input to the model
x1 = torch.randn(10, 64)
