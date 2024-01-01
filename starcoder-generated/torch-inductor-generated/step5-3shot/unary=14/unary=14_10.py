
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=(3, 3), stride=(2, 2))
        self.convt1 = torch.nn.ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.convt1(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 4, 32, 32)
