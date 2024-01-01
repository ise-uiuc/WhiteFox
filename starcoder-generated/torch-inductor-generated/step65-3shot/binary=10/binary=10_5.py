
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 100)
 
    def forward(self, x) {
        v1 = self.linear(x)
        v2 = v1 + other
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 100)
__other__ = torch.randn(1, 100)
