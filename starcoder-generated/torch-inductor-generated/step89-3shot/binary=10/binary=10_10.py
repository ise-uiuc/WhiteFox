
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(256, 128)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()
m.other = torch.nn.Parameter(torch.randn(128, 256))

# Inputs to the model
x1 = torch.randn(1, 256)
