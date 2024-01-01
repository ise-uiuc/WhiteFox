
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, bias=False)
        self.group_norm = torch.nn.GroupNorm(1, 3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.group_norm(v1)
        return v2
# Inputs to the model
x = torch.randn(32, 3, 224, 224)
