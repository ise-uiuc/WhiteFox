
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv3d(2, 2, 3)
        self.conv2 = torch.nn.Conv3d(2, 2, 3)
        self.bn = torch.nn.BatchNorm3d(2)
        self.relu = torch.nn.ReLU()
    def forward(self, x2):
        v2 = self.relu(self.bn(self.conv1(x2)))
        v2 = self.relu(self.bn(self.conv2(v2)))
        return v2
# Inputs to the model
x2 = torch.randn(1, 2, 3, 4, 4)
# Model Ends