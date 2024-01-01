
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(22, 54)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        positive = v1 > 0
        negative_slope = 0.33
        v2 = v1 * negative_slope
        v3 = torch.where(positive, v1, v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(22)
