 description
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
 
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + 0
        v3 = v1 + 1
        v4 = v1 + 2
        v5 = v1 + 0
        v6 = v5 + 1
        v7 = v5 + 2
        v8 = torch.cat((v2, v3, v4, v6, v7), 0)
        return v8

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3)
