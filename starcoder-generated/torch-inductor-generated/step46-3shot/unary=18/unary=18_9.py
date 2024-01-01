
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(1, 8), stride=(1, 7), padding=0)
        self.conv2 = torch.nn.Conv2d(15, 17, kernel_size=5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 10, 35, 167)
