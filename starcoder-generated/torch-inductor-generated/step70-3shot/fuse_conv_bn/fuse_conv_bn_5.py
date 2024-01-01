
# functional API
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        c = torch.nn.Conv2d(2, 4, 3)
        c.weight = torch.nn.Parameter(torch.randn(c.weight.shape))
        c.bias = torch.nn.Parameter(torch.randn(c.bias.shape))
        bn = torch.nn.BatchNorm2d(4)
        bn.track_running_stats = True
        def bn_relu(sub_linear):
            return torch.nn.Sequential(bn, torch.nn.ReLU(inplace=True))
        self.layer = torch.nn.Sequential(c, bn_relu)
    def forward(self, x1):
        v1 = self.layer(x1)
        return v1
x1 = torch.randn(2, 2, 4, 4)
