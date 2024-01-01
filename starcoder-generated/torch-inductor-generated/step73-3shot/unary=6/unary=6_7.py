
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, stride=1, padding=1, bias=False)
    def forward(self, *input):
        t1 = self.conv1(input)
        t2 = 3 + t1
        t3 = t2.clamp(0, 6)
        t4 = t1.mul(t3)
        t5 = t4.div(6)
        return t5
# Inputs to the model
x1 = torch.randn(2, 3, 64, 64)
