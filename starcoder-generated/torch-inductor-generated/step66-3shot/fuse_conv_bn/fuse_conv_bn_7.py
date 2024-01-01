
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(3, 3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(3, 3, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(3, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 4, 4)
