
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(6, 2, (1, 2), stride=(2, 2), padding=(0, 1), bias=False)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        v1 = self.conv2d(x)
        m = self.sigmoid(v1)
        v2 = F.interpolate(m, scale_factor=(2, 2))
        v3 = (m * v2)
        return v3
# Inputs to the model
x = torch.randn(1, 6, 64, 64)
