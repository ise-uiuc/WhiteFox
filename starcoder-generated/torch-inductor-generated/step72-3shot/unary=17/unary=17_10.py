
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, (2, 2), padding=(0, 0), stride=(1, 1), bias=False)
        self.pool = torch.nn.MaxPool2d(3, stride=3, padding=1)
        self.conv1 = torch.nn.ConvTranspose2d(4, 1, (16, 16), stride=(3, 3), padding=(4, 4), output_padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.pool(v1)
        v3 = self.conv1(v2)
        return torch.sigmoid(v3)
# Inputs to the model
x1 = torch.randn(1, 4, 4, 4)
