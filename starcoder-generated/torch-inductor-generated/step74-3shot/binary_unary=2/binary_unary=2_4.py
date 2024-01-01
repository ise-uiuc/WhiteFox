
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 6, stride=6, padding=6, dilation=6)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = F.relu(v2)
        v4 = v3 + torch.randn_like(v3, requires_grad=True)
        return v4
# Inputs to the model
x1 = torch.randn(2, 6, 64, 64)
