
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 17, kernel_size=9, stride=1, padding=8, bias=True)
        self.conv2 = torch.nn.Conv2d(17, 33, 7, 1, 3, bias=False)
        self.conv3 = torch.nn.Conv2d(33, 3, 1, 1, 0, bias=True)
    def forward(self, x1):
        t1 = self.conv3(self.conv2(self.conv1(x1)))
        t2 = torch.sigmoid(t1)
        return t2
# Inputs to the model
x1 = torch.rand(1, 1, 224, 224)
