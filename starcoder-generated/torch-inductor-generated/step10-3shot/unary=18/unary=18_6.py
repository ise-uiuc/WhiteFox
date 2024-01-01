
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=1,  stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1,  stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=48, kernel_size=1,  stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1,  stride=1, padding=1)
        self.flatten = torch.nn.Flatten()
        self.fc = torch.nn.Linear(in_features=16*6*6, out_features=16)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.flatten(v1)
        v3 = self.fc(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
