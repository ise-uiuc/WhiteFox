
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 3, (3, 3), stride=(1, 1), padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=(1, 8), stride=(1, 1), padding=(0, 3))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(self.conv2(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 28, 28)
