
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(128, 128, bias=False)
        self.linear2 = torch.nn.Linear(128, 128, bias=True)
    def forward(self, x1):
        v1 = self.linear1(x1)
        v2 = 3 + v1
        v3 = torch.clamp_min(v2, 0)
        v4 = torch.clamp_max(v3, 6)
        v5 = v1 * v4
        v6 = self.linear2(v5) / 6

        v7 = self.linear2(v1)
        v8 = 3 + v7
        v9 = torch.clamp_min(v8, 0)
        v10 = torch.clamp_max(v9, 6)
        v11 = v7 * v10
        v12 = v11 / 6
        
        return v6 + v12
# Inputs to the model
x1 = torch.randn(1, 128, 128)
