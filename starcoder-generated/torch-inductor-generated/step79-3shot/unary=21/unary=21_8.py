
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size()[0], -1)
        x = self.flat(x)
        x = torch.tanh(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(64, 3, 32, 32)
