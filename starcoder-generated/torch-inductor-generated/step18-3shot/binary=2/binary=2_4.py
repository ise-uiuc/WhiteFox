
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=1, groups=1, bias=False)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = v1 - torch.relu(v1)
        v3 = v2 - torch.tanh(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 1, 64, 64)
