
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.max_pool = torch.nn.AvgPool2d(3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.max_pool(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3) # This relu layer is removed due to the concatenation of the output to a non-ReLU convolution
        v5 = self.conv3(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
