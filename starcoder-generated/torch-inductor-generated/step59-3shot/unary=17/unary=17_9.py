
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(6, 6, (2, 4), groups=6, padding=(1, 2), bias=True, dilation=(3, 5), stride=(3, 3))
        self.conv2 = torch.nn.Conv2d(6, 6, (2, 4), groups=3, bias=False, padding=(1, 2), dilation=(5, 5), stride=(2, 2))
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        v3 = torch.sigmoid(v1 + v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 15, 30)
x2 = torch.randn(1, 3, 10, 15)
