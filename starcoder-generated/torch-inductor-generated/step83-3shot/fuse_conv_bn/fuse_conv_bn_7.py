
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(4, 1, 2)
        self.bn1 = torch.nn.BatchNorm2d(1, track_running_stats=True)
        self.dropout = torch.nn.Dropout(0.5)
        self.conv2 = torch.nn.Conv2d(1, 1, 2)
        self.softmax = torch.nn.Softmax(dim=0)
        self.conv3 = torch.nn.ConvTranspose2d(3, 2, 2)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(10, 4)
    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(self.bn1(x))
        x = self.conv2(x)
        x = self.sigmoid(self.conv3(x))
        x = self.flatten(x)
        x = self.tanh(self.linear(x))
        x = self.softmax(x)
        return x
# Inputs to the model
x3 = torch.randn(1, 4, 4, 4)
