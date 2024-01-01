
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(192, 256, bias=False)
 
    def forward(self, x1, other=None):
        if other is None:
            return self.linear(x1)
        else:
            return self.linear(x1) + other

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(20, 192)
