
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.Linear(5, 1)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = negative_slope * v1
        v4 = torch.where(v2, v1, v3)
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 5)
