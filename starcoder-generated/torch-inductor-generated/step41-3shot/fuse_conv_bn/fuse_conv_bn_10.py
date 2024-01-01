
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(4)
        self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
        torch.manual_seed(13)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=False)
        torch.manual_seed(7)
        self.bn1 = torch.nn.BatchNorm2d(3)
        torch.manual_seed(19)
        self.bn2 = torch.nn.BatchNorm2d(3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
