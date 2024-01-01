
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
 
        self.negative_slope = 0.2
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v3 = torch.lt(v1, 0)
        v4 = self.negative_slope * v1
        v2 = torch.where(v3, v4, v1)
        return v2

# Initializing the model
m = Model()
 
# Input to the model
x1 = torch.randn(1, 16)
