
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        x3 = torch.cat([x1, x2], 0)
        v1 = self.linear(x3)
        v2 = v1 + x2
        return v2

# Initializing the model.
# Specifying "other" to be a tensor "x2"
m = Model()

# Inputs to the model.
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 3)
