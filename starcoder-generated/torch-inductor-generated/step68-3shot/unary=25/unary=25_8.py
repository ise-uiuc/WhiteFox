
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()
        self.linear = torch.nn.Linear(2, 8)
        self.negative_slope = negative_slope
    
    def forward(self, x1):
        v0 = x1.size(0)
        v1 = x1.size(1)
        v2 = self.linear(x1).view(v0, 1, 1, v2)
        v3 = torch.full((v0, v1), self.negative_slope, dtype=v2.dtype, device=v2.device)
        v4 = v2.gt(0)
        v5 = v3.expand_as(v4)
        v6 = torch.where(v4, v2, v5)
        return v6


# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 2)
