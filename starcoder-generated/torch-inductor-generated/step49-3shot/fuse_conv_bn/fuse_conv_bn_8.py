
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3, 5, 7)
        self.bn1 = torch.nn.BatchNorm1d(3)
        self.conv2 = torch.nn.Conv2d(1, 10, 7)
        self.bn2 = torch.nn.BatchNorm2d(1)
    def forward(self, x3, x1):
        y3 = self.conv1(x3.clone())
        y3 = self.bn1(y3.clone())
        y1 = self.conv2(x1.clone())
        y1 = self.bn2(y1.clone())
        return y3.view(-1), y1.view(-1)
# Inputs to the model
x3 = torch.randn(1, 3, 10, 10)
x1 = torch.randn(1, 1, 8, 8)
