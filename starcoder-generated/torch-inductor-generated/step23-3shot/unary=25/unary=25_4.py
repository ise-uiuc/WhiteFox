
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.negative_slope = 0.1
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1.unsqueeze(1)
        v3 = v2 > 0
        v4 = v2 * self.negative_slope
        v5 = torch.where(v3>0, v1, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 128)
