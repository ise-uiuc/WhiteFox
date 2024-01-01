
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(12, 12, 4, stride=2, bias=False)
        self.bn = torch.nn.BatchNorm3d(12, eps=0.129270716508965, running_var=torch.tensor([4.6542, 4.1947, 3.3570, 3.3697]))
    def forward(self, x1):
        v1 = self.conv_transpose(x1) + 3
        v2 = torch.clamp_min(v1, 0)
        v3 = torch.clamp_max(v2, 6)
        v4 = v3 / 6
        v5 = self.bn(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 12, 16, 16, 16)
