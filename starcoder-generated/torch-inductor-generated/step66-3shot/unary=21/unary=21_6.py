
class ModelTanh(torch.nn.Module):
    def __init__(self):
        nn.Module.__init(self)
        self.conv = nn.Conv2d(3, 16, (2, 2), stride=(1, 1), padding=(2, 2), bias=False)
    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
