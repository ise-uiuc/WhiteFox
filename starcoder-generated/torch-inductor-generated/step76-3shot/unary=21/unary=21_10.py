
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 3, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv2 = torch.nn.Conv2d(3, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        return x
# Inputs to the model
x = torch.randn(1, 4, 28, 28)
