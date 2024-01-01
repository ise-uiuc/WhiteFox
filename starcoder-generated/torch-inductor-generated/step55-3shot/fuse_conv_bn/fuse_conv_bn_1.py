
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, 3, bias=False, padding=[1, 0])
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(16, track_running_stats=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        x = self.bn1(self.conv(x))
        x = self.relu(torch.nn.functional.pad(x, [1, 1, 0, 0], value= 1e-05))
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 4, 4)
