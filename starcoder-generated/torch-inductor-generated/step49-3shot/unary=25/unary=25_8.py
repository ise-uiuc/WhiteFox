
class Model(torch.nn.Module):
    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.negative_slope = negative_slope
 
    def forward(self, x1):
        v1 = torch.nn.Linear(8, 6)(x1)
        t2 = v1.flatten() > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(t2.to(dtype=x1.dtype), v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
