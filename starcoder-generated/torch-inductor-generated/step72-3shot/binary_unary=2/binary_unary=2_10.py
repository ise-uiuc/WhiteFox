
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(1, 32, (2, 2), stride=(2, 2), padding=(1, 1))
    def forward(self, x1):
        x = self.conv1(x1).flatten(1)
        x = self.conv2(x1).flatten(1)
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
