
class Model(torch.nn.Module):
    def __init__(self, min_value=-4, max_value=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2, 0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.min_value
        v2 = self.max_value
        x = self.conv1(x)
        x = x.clamp(min=v1, max=v2)
        x = self.conv2(x)
        x = x.clamp(min=v1, max=v2)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.clamp(min=v1, max=v2)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
