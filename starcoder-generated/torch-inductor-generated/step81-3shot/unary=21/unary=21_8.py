
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 7, stride=(2, 1), padding=(3, 2), dilation=(2, 1))
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
