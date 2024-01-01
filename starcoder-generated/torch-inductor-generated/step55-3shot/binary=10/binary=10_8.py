
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 8)
 
    def forward(self, x1, x2):
        v0 = x2.norm().expand(x2.size()[0], x2.size()[1]).permute(0, 2, 1).div(x2.size()[-1]).expand(x2.size())
        v1 = self.linear(x1)
        v2 = v1 + v0
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 4, 3, 1)
x2 = torch.randn(1, 1, 3, 1)
