
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1024, 1024)
 
    def forward(self, x1, x2):
        v1 = self.linear(x1)
        out = v1 + x2
        return out

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(2, 1024)
x2 = torch.randn(2, 1024)
