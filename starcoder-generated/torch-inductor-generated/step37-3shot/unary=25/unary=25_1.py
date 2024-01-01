
negative_slope = [0.3, 0.5, 0.7, 0.9]
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 8)])
 
    def forward(self, x1):
        for i in range(4):
            v1 = self.linears[i](x1)
            v2 = v1 > 0
            v3 = v1 * negative_slope[i]
            v4 = torch.where(v2, v1, v3)
            x1 = v4
        return x1

# Initializing the model
linear_count = 4
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
