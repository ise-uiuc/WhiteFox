
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, padding = 1, dilation = 2)
    def forward(self, x):
        y = self.conv(x)
        torch.onnx.export(self, (x,), "abc.onnx", verbose = False)
        return y
# Inputs to the model
x = torch.randn(1, 3, 12, 12)
