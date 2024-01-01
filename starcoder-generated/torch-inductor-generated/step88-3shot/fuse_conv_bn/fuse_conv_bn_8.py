
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(1, 60, (3, 3, 3), padding=(1, 1, 1), bias=False)
        self.bn1 = torch.nn.BatchNorm3d(60)
        self.conv2 = torch.nn.Conv3d(60, 60, (3, 3, 3), bias=False)
        self.bn2 = torch.nn.BatchNorm3d(60)
        self.conv3 = torch.nn.Conv3d(60, 60, (3, 3, 3), padding=(1, 0, 1), stride=(2, 2, 2), bias=False)
        self.bn3 = torch.nn.BatchNorm3d(60)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        y = self.bn3(x)
        return y
# Inputs to the model
x = torch.randn(2, 1, 16, 16, 16)
