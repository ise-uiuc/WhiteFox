
class Model(torch.nn.Module):
    def __init__(self, other):
        super().__init__()
        self.linear = torch.nn.Linear(80, 64)
        self.__constant_var0 = torch.tensor(6, dtype=torch.long)
        self.other = other
 
    def forward(self, x):
        v1 = self.linear(x).size(1)
        v2 = self.__constant_var0
        v3 = v1 - v2
        v5 = v3 + self.other
        return v5

# Initializing the model
o = torch.randint(100)
m = Model(o)

# Inputs to the model
x = torch.randn(1, 80)
