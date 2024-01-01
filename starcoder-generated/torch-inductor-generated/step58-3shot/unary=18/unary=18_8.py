
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        return nn.ReLU6()(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
