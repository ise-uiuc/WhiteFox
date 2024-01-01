
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(3,3,1)
        self.bn = torch.nn.BatchNorm3d(3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv3d(3,3,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
# Inputs to the model
x = torch.randn(1,3,3,3,3)
