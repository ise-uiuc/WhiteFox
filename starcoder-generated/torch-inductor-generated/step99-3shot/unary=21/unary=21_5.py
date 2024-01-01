
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 8, 3, stride=(2, 1), padding=[1, 0, 1, 2])
    def forward(self, x):
        t1 = self.conv(x)
        t2 = torch.tanh(t1)
        return t2
# Inputs to the model
x = torch.randn(1, 5, 24, 24)
