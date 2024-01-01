
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 5, stride=(1, 1), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(64,8, 5, stride=(1, 2), padding=(0, 1))
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
