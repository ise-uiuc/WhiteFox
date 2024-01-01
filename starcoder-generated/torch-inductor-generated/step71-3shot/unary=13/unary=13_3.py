
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.v1 = torch.nn.Linear(32, 64, bias=False)
        self.v2 = torch.nn.Linear(64, 16, bias=False)
 
    def forward(self, x1):
        v1 = self.v1(x1)
        v2 = self.v2(v1)
        v3 = torch.sigmoid(v1)
        v4 = v2 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32)
