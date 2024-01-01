
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 7, 2, padding=2)
        self.conv2 = torch.nn.Conv2d(7, 9, 2)
    def forward(self, x1):
        a1 = self.relu1(self.conv1(x1))
        a2 = self.relu2(self.conv2(a1))
        a3 = a2 - 3
        a4 = F.relu6(a3)
        return a4
# Inputs to the model
x1 = torch.randn(1, 5, 224, 224)
