
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d_3 = torch.nn.Conv2d(226, 500, 3, stride=1, padding=0)
        self.conv2d_5 = torch.nn.Conv2d(500, 1000, 3, stride=1, padding=0)
        self.conv2d_7 = torch.nn.Conv2d(1000, 606, 2, stride=1, padding=0)
        self.transposeconv2d_10 = torch.nn.ConvTranspose2d(606, 400, 6, stride=9, padding=0)
    def forward(self, x1):
        v1 = self.conv2d_3(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        v4 = self.conv2d_5(v3)
        v5 = torch.sigmoid(v4)
        v6 = v4 * v5
        v7 = self.conv2d_7(v6)
        v8 = torch.sigmoid(v7)
        v9 = v1 * v4
        v10 = self.transposeconv2d_10(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 226, 10, 10)
