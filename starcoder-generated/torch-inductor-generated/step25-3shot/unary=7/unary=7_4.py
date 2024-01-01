
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
    def clip56(self, x1):
        v1 = x1 + 3
        v2 = v1.clamp(0, 6)
        return v2
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = self.clip56(v1) * 6
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
