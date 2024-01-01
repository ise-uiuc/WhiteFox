
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 5, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 20, 5, stride=(1, 2), padding=(2, 3))
        self.conv3 = torch.nn.ConvTranspose2d(20, 10, (1, 4), stride=(3, 1), padding=(0, 3))
    def forward(self, x1):
        v1 = self.conv3(self.conv2(self.conv1(x1)))
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
