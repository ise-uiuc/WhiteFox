
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=0, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.sigmoid(v1)
        f1 = torch.tensor(range(16))
        f2 = f1[None, :, None, None]*1./16
        v3 = v1 * f2
        return v3
# Inputs to the model
x1 = torch.tensor(range(16)).reshape(1, 1, 4, 4)
