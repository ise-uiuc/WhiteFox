
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=(1, 3), stride=(2, 2), padding=(1, 2), bias=None)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
input = torch.randn(1, 1, 127, 127)
