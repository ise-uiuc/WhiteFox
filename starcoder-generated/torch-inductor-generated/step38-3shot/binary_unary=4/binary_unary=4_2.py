
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16)
 
    def forward(self, x1, other=0):
        x1 = self.linear(x1)
        x2 = x1 + other
        x3 = torch.nn.functional.gelu(input=x2, approximate=True)
        return x3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
