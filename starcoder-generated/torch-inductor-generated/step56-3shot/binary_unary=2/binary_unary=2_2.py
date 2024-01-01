
class MyModel(torch.nn.Module):
    def __init__(self, hidden=True):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        if hidden:
            v5 = v1 - 10
        else:
            v5 = torch.nn.functional.upsample_blanc_zeropad2d(v1, scale_factor=2, align_corners=False)
        v6 = F.relu(v5)
        v4 = v6.unsqueeze(0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 16, 56, 56)
