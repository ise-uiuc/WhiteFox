
class Model(torch.nn.Module):
    def __init__(self):
        self.other = torch.arange(8).reshape(1, 8)
        super().__init__()
        self.linear = torch.nn.Linear(10, 64)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 10)
