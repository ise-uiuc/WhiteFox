
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 7), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(3, 7), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(3, 7), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 96, 96)
