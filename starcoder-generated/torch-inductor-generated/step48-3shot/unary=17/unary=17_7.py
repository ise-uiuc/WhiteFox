
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, kernel_size=3, stride=3)
        self.pool = torch.nn.AvgPool2d(3, stride=3)
        self.fc1 = torch.nn.Linear(3264, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 3)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.pool(v1)
        v3 = torch.flatten(v2, 1)
        v4 = self.fc1(v3)
        v5 = torch.relu(v4)
        v6 = self.fc2(v5)
        v7 = torch.relu(v6)
        v8 = self.fc3(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 240, 320)
