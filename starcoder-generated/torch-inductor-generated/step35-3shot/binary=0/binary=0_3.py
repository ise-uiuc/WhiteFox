
class Model(torch.nn.Module):
    def __init__(self):
       super(Model, self).__init__()
       self.pool = torch.nn.MaxPool2d( kernel_size=2, stride=2, padding=1, dilation=2)
    def forward(self, x1, other=None):
        v1 = self.pool(x1)
        if other == None:
            other = torch.randn(v1.shape)
        v2 = v1 + other
        return v2
# Inputs to the model
x1 = torch.randn(15, 19, 19, 31)
