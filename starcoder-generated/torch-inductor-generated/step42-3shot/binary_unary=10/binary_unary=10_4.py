
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m = Linear(3, 3)
        self.other = torch.randn(3, 3)
 
    def forward(self, x1):
        v1 = self.m(x1)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
