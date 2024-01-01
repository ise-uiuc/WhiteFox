
nn1 = torch.nn.Linear(3, 6)
nn2 = torch.nn.Sigmoid()
nn3 = torch.nn.Linear(3, 6)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn1
        self.sigmoid = nn2
        self.linear = nn3
 
    def forward(self, x1, x2):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.linear(x2)
        v5 = self.sigmoid(v4)
        v6 = v4 * v5
        return torch.cat((v3, v6), 0)

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(6, 3, 64, 64)
