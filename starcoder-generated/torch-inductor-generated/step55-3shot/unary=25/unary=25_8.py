
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.linear = torch.nn.Linear(15, 30)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = torch.gt(v1, 0).float()
        v3 = 0.2859967 * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model(0.2859967)

# Inputs to the model
x1 = torch.randn(128, 15)
