
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x2):
        v2 = self.linear(x2)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(16, 8)
