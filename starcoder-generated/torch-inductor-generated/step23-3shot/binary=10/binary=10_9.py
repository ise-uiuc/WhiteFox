
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.const = torch.randn(1)*20.0
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.const
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 16)
