
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.relu_1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 32, 1, stride=1, padding=0)
        self.relu_2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(32, 1, 3, stride=1, padding=1)
        self.relu_3 = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_1(x)
        x = self.conv2(x)
        x = self.relu_2(x)
        x = self.conv3(x)
        x = self.relu_3(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 55, 55)
