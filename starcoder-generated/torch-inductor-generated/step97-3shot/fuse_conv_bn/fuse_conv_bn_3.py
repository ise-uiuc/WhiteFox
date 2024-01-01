
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)
        self.conv1 = torch.nn.Conv2d(64, 32, (3, 3), stride=1, padding=(1, 1), bias=False)
        self.bn2 = torch.nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True)
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        return torch.nn.functional.relu(x)
# Inputs to the model
x = torch.randn(2, 64, 14, 14)
