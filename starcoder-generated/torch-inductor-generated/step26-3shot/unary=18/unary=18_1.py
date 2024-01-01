
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 9, (1, 3), stride=(1, 2), padding=(0, 1))
        self.maxpool = torch.nn.MaxPool2d((1, 3), stride=(1, 3), padding=(0, 1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.maxpool(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 64)
