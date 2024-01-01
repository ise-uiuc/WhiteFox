
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 > 0
        v3 = v2.float()
        v4 = v1 * 0.1
        v5 = torch.where(v2, v1, v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
