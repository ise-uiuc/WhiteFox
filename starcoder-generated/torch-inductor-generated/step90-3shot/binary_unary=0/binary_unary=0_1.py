
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(147, 147, 3, stride=2)
        self.conv2 = torch.nn.Conv2d(147, 147, 5, stride=2)
        self.conv3 = torch.nn.Conv2d(147, 147, 7, stride=2)
        self.conv4 = torch.nn.Conv2d(147, 147, 31, stride=2)
        self.conv5 = torch.nn.Conv2d(147, 147, 101, stride=2)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        return conv5
# Inputs to the model
x = torch.randn(1, 147, 233, 233)
