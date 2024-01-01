
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        vx = v1 * -0.1
        v2 = v1 > 0
        v3 = torch.where(v2, v1, vx)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
