
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(3, 9, 9, stride=1, padding=0)
    def forward(self, x1):
        y = self.conv(x1)
        z = self.relu(y)
        return z

