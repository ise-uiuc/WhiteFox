
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 4)
 
    def forward(self, x1):
        v3 = self.linear(x1)
        v2 = v3 - 2.4
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 64)
