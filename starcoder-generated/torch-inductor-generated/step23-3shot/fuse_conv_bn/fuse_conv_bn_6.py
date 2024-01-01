
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 3, padding=0)
        self.conv2 = torch.nn.Conv2d(6,8, 3, padding=0)
        self.conv3 = torch.nn.Conv2d(8, 9, 3, padding=0)
        self.conv4 = torch.nn.Conv2d(9, 10, 3, padding=0)
    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        return self.conv4(x1)
# Input to the model
x1 = torch.ones(1, 3, 224, 224)
