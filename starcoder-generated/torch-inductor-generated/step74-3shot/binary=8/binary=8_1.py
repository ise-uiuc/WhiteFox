
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(3, stride=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 10, 3, stride=2, padding=1)
        self.pool2 = torch.nn.MaxPool2d(2, stride=2, padding=1)

    def forward(self, img):
        conv1 = self.pool1(self.conv1(img))
        conv2 = self.pool2(self.conv2(conv1))
        return conv2
# Inputs to the model
img = torch.randn(1, 3, 64, 64)
