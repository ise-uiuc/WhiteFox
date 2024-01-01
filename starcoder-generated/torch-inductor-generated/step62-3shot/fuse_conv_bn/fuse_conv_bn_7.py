
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 20, 5, 1, 2)
        self.conv2 = torch.nn.Conv2d(50, 64, 5, 1, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 128, 128)
