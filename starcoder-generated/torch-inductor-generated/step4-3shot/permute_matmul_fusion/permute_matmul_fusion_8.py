
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv0D(1, 1, 1)
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v3 = x2.permute(0, 2, 1)
        v2 = torch.bmm(v1, v3)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 1)
x2 = torch.randn(1, 1, 1)
