
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main_encoder = torch.nn.Sequential()
        self.main_encoder.add_module('layers0', torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1, 1, 4)))
        self.main_encoder.add_module('layers1', torch.nn.Sequential(torch.nn.ReLU(True)))
    def forward(self, x):
        o = self.main_encoder(x)
        o = o + o
        o = o + o
        o = o + o
        return o
# Inputs to the model
x1 = torch.randn(3, 3, 1, 1)
