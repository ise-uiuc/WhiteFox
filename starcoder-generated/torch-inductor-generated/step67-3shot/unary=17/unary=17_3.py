
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose1d(in_channels=49, out_channels=49, kernel_size=(2, 2), stride=(2, 2), bias=False)
        self.conv2 = torch.nn.ConvTranspose1d(1, 1, kernel_size=(2, 2), stride=(2, 2), bias=False)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        return x1
# Inputs to the model
x = torch.randn(1, 49, 36)
