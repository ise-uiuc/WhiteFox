
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        return v2
# Inputs to the model
x = torch.randn(1, 64, 11, 42)
