
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, stride=1, padding='valid')
    def forward(self, x):
        x = self.conv1(x)
        return x
