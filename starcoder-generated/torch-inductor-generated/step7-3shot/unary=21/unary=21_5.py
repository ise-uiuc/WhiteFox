
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), torch.nn.Conv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    def forward(self, x):
        y1 = self.conv1(x)
        y2 = torch.nn.Tanh()(y1)
        y3 = self.conv2(y2)
        y4 = torch.nn.Tanh()(y3)
        return y4
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
