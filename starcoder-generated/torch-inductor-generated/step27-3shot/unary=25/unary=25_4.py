
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.Linear(64, 8)
        self.negative_slope = 0.1
 
    def forward(self, x1):
        v1 = self.ln1(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
