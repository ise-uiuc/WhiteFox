
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.gt(v1, 0).float()
        v3 = -1 / 100 * v2
        v4 = torch.where(v2, v1, v3)
        
        return v4

# Initializing the model
m = Model(-3.5)

# Inputs to the model
x1 = torch.randn(1, 8)
