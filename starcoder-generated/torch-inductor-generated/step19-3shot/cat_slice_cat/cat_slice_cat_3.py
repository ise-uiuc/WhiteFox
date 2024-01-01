
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 256, 3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.max_pool1 = torch.nn.MaxPool3d(2, stride=2)

    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu1(v1)
        v3 = self.conv2(v2)
        v4 = self.relu2(v3)
        v5 = self.max_pool1(v4)
        v6 = torch.flatten(v5, start_dim=1)
        return v6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
