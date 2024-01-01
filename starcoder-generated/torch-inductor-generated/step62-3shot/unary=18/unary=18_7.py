
class Model(nn.Module):
    def __init__(self, m1, m2):
        super().__init__()
        if m1 <= m2:
            self.conv1 = torch.nn.Conv2d(in_channels=m1, out_channels=m2, kernel_size=3, stride=1, padding=1)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=m2, out_channels=m1, kernel_size=3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
m1 = torch.randint(low=3, high=1000, size=(1,))
m2 = torch.randint(low=3, high=1000, size=(1,))
x1 = torch.randn(1, m1, 224, 224)
