
class Module1(torch.nn.Module):
    def __init__(self, dim, bias):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim, bias=bias)
 
    def forward(self, x):
        v2 = self.linear(x)
        return v2
 
class Model(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.module1 = Module1(num_features, False)
        self.module2 = Module1(num_features, True)
 
    def forward(self, x):
        v1 = self.module1(x)
        v2 = self.module2(x)
        res = v2 + v1
        return res

# Initializing the model
num_features = 32
m = Model(num_features)

# Inputs to the model
x = torch.randn(2, num_features)
