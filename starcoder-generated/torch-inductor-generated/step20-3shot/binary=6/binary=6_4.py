
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 16, bias=False)
 
    def forward(self, x1):
        v1 = self.linear(x1)
        v2 = v1 - 1
        v3 = v2[0][0]
        v4 = v2[1][2]
        v5 = v3 + v4
        v6 = v5 + 2
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8)
