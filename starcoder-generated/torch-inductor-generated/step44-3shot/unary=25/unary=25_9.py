
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(12544, 25088)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        negative_slope = 0.2
        v3 = negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 25088)
