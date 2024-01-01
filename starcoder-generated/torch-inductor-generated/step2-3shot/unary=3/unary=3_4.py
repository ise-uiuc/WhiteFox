
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_gelu = torch.nn.Conv1d(3, 2, 1, stride=1, padding=0, groups=1)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv_gelu(x1)
        v2 = self.gelu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 8)
