
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv(1, 1, 2)
        self.bn = torch.nn.BatchNorm2d(1)

    def init_parameters(self, conv, bn):
        self.conv.weight = conv.weight
        self.conv.bias = conv.bias
        self.bn.weight = bn.weight
        self.bn.bias = bn.bias
        self.bn.running_mean = bn.running_mean
        self.bn.running_var = bn.running_var

    def forward(self, x):
        x1 = self.conv(x)
        x2 = torch.nn.functional.batch_norm(x1, self.bn.running_mean, self.bn.running_var, self.bn.weight, self.bn.bias, training=False, eps=self.bn.eps)
        return x2

# Initializing the model
m = Model()
conv = torch.nn.Conv2d(1, 1, 2)
bn = torch.nn.BatchNorm2d(1)
conv(torch.randn(1, 1, 2, 2))
bn(conv(torch.randn(1, 1, 2, 2)))
m.init_parameters(conv, bn)
m1 = torch.jit.script(m)

# Inputs to the model
x = torch.randn(1, 1, 2, 2)
