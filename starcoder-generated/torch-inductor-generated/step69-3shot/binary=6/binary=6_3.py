
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8, bias=False)
 
    def forward(self, x1, other):
        x = x1.clone()
        x.__iadd__(other)
        y = self.linear(x)
        y.__isub__(other)
        return y

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 8)
other = torch.randn(1)
