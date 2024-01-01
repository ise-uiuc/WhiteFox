
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
 
    def forward(self, x1):
        v1 = self.pool(self.relu(self.conv1(x1)))
        v2 = self.relu(self.conv1(self.relu(self.pool(v1))))
        v3 = torch.cat([v1, v2], dim=1)
        v4 = v3[:, 0:4194303]
        return v4

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
