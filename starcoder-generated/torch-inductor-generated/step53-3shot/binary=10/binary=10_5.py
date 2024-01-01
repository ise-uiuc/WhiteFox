
class Model(torch.nn.Module):
    __other__ = torch.randn(3, 3, dtype=torch.float)
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, v1):
        o1 = self.linear(v1)
        o2 = o1 + self.__other__
        return o2

# Initializing the model
m = Model()

# Inputs to the model
v1 = torch.randn(2, 2)
