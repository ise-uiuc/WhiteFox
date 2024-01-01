
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(16, 16, 3, stride=1, padding=1),
             torch.nn.Conv2d(16, 16, 1, stride=1, padding=0)])
    def forward(self, x1, x2):
        v2 = x1 + x2
        v3 = torch.relu(v2)
        v4 = self.convs[0](v3)
        v5 = self.convs[1](v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = 1
