
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 2)
        self.conv2 = torch.nn.Conv2d(64, 32, 2)
        self.conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.relu = torch.nn.ReLU()
        self.conv5 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(32, 32, 3, padding=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)
        return torch.tanh(x)
# Inputs to the model
x = torch.randn(1, 2, 64, 64)
