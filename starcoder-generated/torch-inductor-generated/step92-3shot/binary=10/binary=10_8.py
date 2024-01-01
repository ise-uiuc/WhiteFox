
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other
 
    def forward(self, x1):
        v1 = F.linear(x1, self.other)
        v2 = v1 + self.other
        return v2

# Initializing the model
one_tensor = torch.rand(10)
m = Model(one_tensor)

# Inputs to the model
x1 = torch.randn(1, 10)
