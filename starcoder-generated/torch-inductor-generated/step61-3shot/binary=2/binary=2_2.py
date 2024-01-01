
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv0 = torch.nn.Conv2d( 512, 48, kernel_size=(1, 1), stride=(1, 1))
    def forward(self, x):
        v1 = self.conv0(x)
        v2 = v1 - 0.31
        return v2
# Inputs to the model
x = torch.randn(1, 512, 600, 1000)
