
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
 
    def forward(self, x2, x3):
        v1 = torch.cat([x2, x3], dim=1)
        v2 = v1[:, 0:(2**63 - 1)]
        v3 = v2[:, 0:1]
        v4 = torch.cat([v1, v3], dim=1)
        v5 = self.relu6(v4)
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x2 = torch.randn(1, 32, 128, 128)
x3 = torch.randn(1, 32, 128, 128)
