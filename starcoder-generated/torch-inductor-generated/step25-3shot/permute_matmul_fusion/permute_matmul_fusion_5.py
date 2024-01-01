
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.functional.relu
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], -1)
        v1 = self.relu(torch.transpose(x, -1, -2)).permute(0, 2, 1, 3)
        v2 = self.relu(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        v3 = v1 + v2
        v4 = torch.bmm(v1, v2)
        v5 = torch.bmm(x1.permute(0, 2, 1), x2)
        v6 = x2.permute(0, 2, 1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
x2 = torch.randn(1, 2, 2, 2)
