
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
 
    def forward(self, x2):
        v3 = self.linear(x2)
        v4 = v3 > 0
        v5 = v3 * 0.1
        v6 = torch.where(v4, v3, v5)
        return v6

# Initializing the model
d = Model()

# Inputs to the model
x2 = torch.randn(3, 64)
