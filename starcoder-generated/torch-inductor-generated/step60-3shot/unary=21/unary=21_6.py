
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2D_4 = torch.nn.Conv2d(120, 84, kernel_size=(5, 1), stride=(1, 1), groups=1)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        y = self.conv2D_4(x)
        y = self.tanh(y)
        return y
# Inputs to the model
x = torch.randn(1, 120, 13, 4)
