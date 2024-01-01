
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
        self.negative_slope = 0.35
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        v5 = self.linear(x2)
        v6 = v5 > 0
        v7 = v5 * self.negative_slope
        v8 = torch.where(v6, v5, v7)
        return torch.cat([v4, v8])
        
# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2, 8)
x2 = torch.randn(1, 2, 8)

