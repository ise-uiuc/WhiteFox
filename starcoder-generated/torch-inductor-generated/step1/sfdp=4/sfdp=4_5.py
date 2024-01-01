
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.batch = torch.nn.BatchNorm2d(12)
        self.relu0 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(4*4*12, 256)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(256, 256)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(256, 1)
 
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = self.relu0(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        v1 = self.fc3(x)
        return v1

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
