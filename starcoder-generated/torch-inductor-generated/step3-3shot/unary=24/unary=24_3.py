
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        m1 = v1 > 0
        m2 = m1.squeeze(1)
        m3 = m2.unsqueeze(1)
        v2 = v1 + m3
        v3 = torch.where(m1, v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
