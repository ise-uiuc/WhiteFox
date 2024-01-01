
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
        torch.manual_seed(1)
        self.norm1 = torch.nn.BatchNorm2d(3, track_running_stats=True)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
        torch.manual_seed(1)
        self.norm2 = torch.nn.BatchNorm2d(3, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x

# Inputs to the model
x = torch.randn(1, 3,63, 63)
