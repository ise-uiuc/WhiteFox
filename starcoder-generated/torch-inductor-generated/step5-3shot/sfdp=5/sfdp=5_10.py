
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(128, 64)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        v2 = self.linear(x2)
        v3 = v1 @ v2.transpose(-2, -1)
        v4 = v3 / math.sqrt(v3.size(-1))
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 128)
x2 = torch.randn(2, 128)
