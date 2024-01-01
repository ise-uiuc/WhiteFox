
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v12 = []
        v11 = []
        v4 = []
        v3 = x1
        v18 = self.linear(v3)
        v5 = v18.permute(0, 2, 1)
        v12.append(v5)
        v10 = torch.cat(v12, 0)
        v14 = torch.cat(v11, 0)
        v13 = torch.cat(v4, 0)
        return v10
# Inputs to the model
x1 = torch.randn(1, 2, 2)
