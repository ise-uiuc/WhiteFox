
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv1 = torch.nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(1, 16), stride=(1, 1),
                                              padding=(0, 8))
    def forward(self, x1):
        v1 = self.maxpool2(x1)
        v2 = self.relu(v1)
        v3 = self.conv1(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 64, 256)
