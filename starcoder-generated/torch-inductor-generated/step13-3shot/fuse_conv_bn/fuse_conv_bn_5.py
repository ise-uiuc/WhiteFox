
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 2)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
