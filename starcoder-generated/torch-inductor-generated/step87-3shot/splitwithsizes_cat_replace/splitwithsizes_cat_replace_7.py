
from collections import OrderedDict
class Block(torch.nn.Module):
    def __init__(self, inp, hidden):
        super().__init__()
        self.conv0 = torch.nn.Sequential(OrderedDict([('dropout0', torch.nn.Dropout(p=0.25)), ('dropout1', torch.nn.Dropout2d(p=0.5)), ('dropout2', torch.nn.Dropout3d(p=0.75))]))
        self.conv1 = torch.nn.Conv2d(inp * 2, hidden, 1, 1, bias=False)
        self.conv2 = torch.nn.Conv2d(hidden, hidden * 2, 3, 1, 0, bias=False)
        self.conv3 = torch.nn.Conv2d(hidden, hidden * 4, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(hidden, affine=False, track_running_stats=False)
        self.bn2 = torch.nn.BatchNorm2d(hidden * 2, affine=False, track_running_stats=False)
        self.bn3 = torch.nn.BatchNorm2d(hidden * 4, affine=False, track_running_stats=False)
    def forward(self, x1):
        out = torch.nn.ReLU()(self.bn1(self.conv1(self.conv0(x1))))
        # The "out" tensor below will also be used in an add operation, and
        # Pytorch has its own mechanism to identify if it is in training mode.
        # Hence, please set torch.nn.Dropout to eval mode when generating the
        # script. If the operation is part of training, there could be some
        # problem for the pattern to be triggered.
        return torch.nn.ReLU()(self.bn2(self.conv2(out))) + torch.nn.Sigmoid()(self.bn3(self.conv3(out)))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Block(3, 16)
    def forward(self, v1):
        out = self.features(v1)
        return (v1 - out, out)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
