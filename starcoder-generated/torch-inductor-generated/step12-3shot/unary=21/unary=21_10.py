
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        x1 = torch.tanh(x)
        y = self.conv1(x1)
        y = self.conv2(y)
        return y.detach()
# Inputs to the model
x = torch.randn(64, 3, 64, 64)
