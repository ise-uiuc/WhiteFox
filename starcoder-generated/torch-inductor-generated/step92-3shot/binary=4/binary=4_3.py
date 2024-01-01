
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.other = torch.randn(32, 32, 3, 3)
        self.linear = torch.nn.Linear(32, 32)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(32, 7, 7)
