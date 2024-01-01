
class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv = torch.nn.Conv2d(4, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        return F.Sigmoid()(v1)
# Inputs to the model
x1 = torch.randn(1, 4, 3, 3)
