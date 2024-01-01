
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
        self.bn1 = torch.nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x1 = self.bn1(x1)
        x1 = self.relu(self.conv2(x1))
        return x1
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
