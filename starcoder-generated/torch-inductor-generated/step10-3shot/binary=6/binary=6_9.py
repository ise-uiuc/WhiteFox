
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)
 
    def forward(self, x3):
        v3 = self.linear(x3)
        v40 = torch.randn(16, 8)
        v4 = v3 - v40
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x3 = torch.randn(16, 8)
