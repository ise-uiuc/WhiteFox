
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(4, 4, 4)
        self.bn = torch.nn.BatchNorm3d(4)
        self.conv2 = torch.nn.Conv3d(1, 1, 1)
        self.bn1 = torch.nn.BatchNorm3d(4)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.relu(self.bn1(x1))
        x1 = torch.transpose(x1, 1, 2)
        x1 = torch.bmm(x1.unsqueeze(1), x1.unsqueeze(2))
        x1 = torch.cat([x1.flatten(1), x1.flatten(0)], -1)
        x2 = self.conv2(x1)
        x1 = self.bn(x2)
        return x1, x2
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
