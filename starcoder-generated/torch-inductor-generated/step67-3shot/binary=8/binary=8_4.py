
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = torch.nn.Sequential(OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)),
            ('conv2', torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)),
            ('conv3', torch.nn.Conv2d(3, 16, 3, stride=1, padding=1))
        ]))
    def forward(self, x1, x2, x3):
        v1 = self.layers(x1)
        v2 = self.layers(x2)
        v3 = self.layers(x3)
        v5 = v1 + v2
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
x3 = torch.randn(1, 3, 32, 32)
