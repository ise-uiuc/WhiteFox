
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(21)
        self.layer1 = torch.nn.Conv2d(3, 3, 1, bias=False)
        torch.manual_seed(1)
        self.layer2 = torch.nn.BatchNorm3d(3)
        torch.manual_seed(2)
        self.layer3 = torch.nn.Conv2d(3, 3, 1, bias=False)
        torch.manual_seed(1)
        self.layer4 = torch.nn.BatchNorm3d(3)
    def forward(self, x1):
        s1 = self.layer1(x1)
        s1 = self.layer2(s1)
        s1 = self.layer3(s1)
        s1 = self.layer4(s1)
        x1 = s1 + s1
# Inputs to the model
x1 = torch.randn(1, 3, 3, 3)
