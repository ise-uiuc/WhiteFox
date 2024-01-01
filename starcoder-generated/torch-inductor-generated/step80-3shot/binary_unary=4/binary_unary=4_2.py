
class Model(torch.nn.Module):
    def __init__(self, __arg1__):
        super().__init__()
        self.linear = torch.nn.Linear(D1 + D2, D3)
        self.other = __arg1__ 
 
    def forward(self, x1):
        v1 = self.linear(x1, self.other)
        v2 = v1 + self.other
        v3 = torch.relu(v2)
        return v3

# Initializing the model
D1 = 5
D2 = 10
D3 = 15
m = Model(torch.randn(D2))

# Inputs to the model
x1 = torch.randn(1, D1 + D2)
