
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1)
        self.conv_avg = torch.nn.Conv2d(128, 8, 8, stride=8, padding=0)
        self.conv_last = torch.nn.Conv2d(8, 64, 1, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2a = v1.mean(dim=[2, 3], keepdim=True)
        v2 = v2a + 1
        v3 = self.conv_avg(v2)
        v4 = F.relu(v3)
        v5 = self.conv_last(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
