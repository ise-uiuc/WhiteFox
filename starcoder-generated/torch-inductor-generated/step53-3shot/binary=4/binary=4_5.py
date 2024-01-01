
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(32, 64, bias=True)
 
        # This is a keyword argument. We will specify it during model generation after initialization.
        self.other = torch.rand(1, 8, 32, 32)

    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 + self.other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 32, 32)
__other__ = torch.rand(1, 8, 32, 32)
