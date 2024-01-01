
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16)
        self.__other__ = other
 
    def forward(self, x1):
        v0 = self.__other__.view(1, 16)
        v1 = self.linear(x1)
        v2 = v1 + v0
        return v2

# Initializing the model
m = Model(torch.randn(1, 16))

# Inputs to the model
x1 = torch.randn(1, 16)
