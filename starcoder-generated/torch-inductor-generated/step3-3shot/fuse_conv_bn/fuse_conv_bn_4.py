
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.cat([x1, x1, x1, x1], 1)
        conv1 = torch.nn.Conv2d(4, 4, 1)
        x2 = conv1(x2)
        bn_module = torch.nn.BatchNorm2d(4)
        bn_module.running_mean = torch.ones_like(bn_module.bias, dtype=torch.float32)
        bn_module.running_var = torch.ones_like(bn_module.bias, dtype=torch.float32)
        x2 = bn_module(x2)

        conv2 = torch.nn.Conv2d(4, 1, 1)
        x2 = conv2(x2)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
