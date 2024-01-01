
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
 
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, other=self.other,...)
        return v1

# Initializing the model
other = torch.randn(1, 50)
m = Model(other)

# Inputs to the model
x1 = torch.randn(1, 50)
