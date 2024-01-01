
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=(2, 1), padding=(2, 1))
        self.conv2 = torch.nn.Conv2d(16, 16, 5, stride=(2, 1), padding=(1, 2))
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=(1, 1), padding=0)
    def forward(self, x3):
        v1 = torch.sigmoid(self.conv1(x3))
        v2 = torch.sigmoid(self.conv2(v1))
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x3 = torch.randn(53, 16, 50, 66)
