
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.view(x1.size())
        v2 = x1.view(x1.size())
        v3 = self.x2.view(x2.size())
        v4 = x2.view(x2.size())
        v5 = v1 + v2
        v6 = torch.relu(v5)
        v7 = v3 + v4
        v8 = torch.relu(v7)
        v9 = v6.to(v8)
        v10 = torch.relu(v9)
        v11 = v8 + v10
        v12 = torch.relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(16, 32, 28, 28)
x2 = torch.randn(16, 32, 28, 28)
