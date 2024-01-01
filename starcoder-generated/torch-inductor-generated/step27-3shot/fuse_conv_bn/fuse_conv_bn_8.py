
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU(inplace=True)
    def forward(self, x2):
        x = self.conv2(self.conv1(x2))
        x = self.relu(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.relu(x)
        return x[0]
# Inputs to the model
x2 = torch.randn(1, 3, 5, 5)
