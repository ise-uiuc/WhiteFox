
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(1, 1, 2)
        self.conv_2 = torch.nn.ConvTranspose2d(1, 1, 2)
        self.conv_3 = torch.nn.Conv2d(1, 1, 2)
        self.bn_1 = torch.nn.BatchNorm1d(17, momentum=0.8)
        self.avg_pool = torch.nn.AvgPool2d((2, 1), stride=1)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = torch.relu(v3)
        v5 = self.bn_1(v4)
        v6 = self.avg_pool(v5)
        v7 = self.softmax(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
