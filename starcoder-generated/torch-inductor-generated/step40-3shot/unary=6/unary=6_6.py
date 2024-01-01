
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 64, 1, stride=1, padding=0)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = self.avgpool(t1)
        t3 = self.relu(t2)
        t4 = t3.squeeze(-1)
        t5 = t4 + 3
        t6 = torch.clamp_min(t5, 0)
        t7 = torch.clamp_max(t6, 6)
        t8 = t4 * t7
        t9 = t8 / 6
        return t9
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
