
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(25, 8, 3, padding=1, stride=2)
        self.conv2 = torch.nn.Conv2d(8, 17, 3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(17, 19, 3, padding=1, stride=2)
        self.conv4 = torch.nn.Conv2d(19, 32, 3, padding=1, stride=2)
        self.conv5 = torch.nn.Conv2d(32, 48, 7, padding=3, stride=1)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(9)
    def forward(self, input_tensor):
        x1 = self.conv1(input_tensor)
        x2 = self.relu(x1)
        x3 = self.pool(x2)
        x4 = self.conv2(x3)
        x5 = self.relu(x4)
        x6 = self.pool(x5)
        x7 = self.conv3(x6)
        x8 = self.relu(x7)
        x9 = self.pool(x8)
        x10 = self.conv4(x9)
        x11 = self.relu(x10)
        x12 = self.pool(x11)
        x13 = self.conv5(x12)
        x14 = self.relu(x13)
        x15 = self.pool(x14)
        return x15
# Input to the model
input_tensor = torch.randn(1, 25, 128, 128)
