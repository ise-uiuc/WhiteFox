
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        negative_slope = 0.5
        v1 = torch.where(v1 > 0, v1, v1 * negative_slope)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
