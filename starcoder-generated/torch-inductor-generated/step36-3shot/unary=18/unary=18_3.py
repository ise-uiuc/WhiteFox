
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=7, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=4, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.sigmoid(v5)
        v7 = self.conv4(v6)
        # Add code for layer 7 here.
        v8 = torch.nn.functional.relu(v7)
        v9 = self.conv5(v8)
        v10 = torch.sigmoid(v9)
        return v10
# Inputs to the model
x1 = torch.randn(1, 6, 64, 64)
