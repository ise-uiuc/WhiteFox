
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3)
        self.conv1 = torch.nn.Conv2d(1, 1, 3)
        self.bn = torch.nn.BatchNorm2d(1)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.bn1 = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        x = self.conv(x)
        x1 = self.conv(x)
        x = self.leaky(x)
        x = self.conv1(x)
        return x
x = torch.randn(3, 3, 224, 224)
model = Model()
graph = torch.fx.symbolic_trace(model)

# Graph begins
graph_str = str(graph)

# Inputs to the model
x = torch.randn(1, 3, 224, 224)


model = nn.BatchNorm2d(num_features=4)
bn = torch.quantization.fuse_modules(torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8), [['weight', 'bias']])
bn = torch.quantization.fuse_modules(torch.quantization.quantize_dynamic(bn, {nn.Conv2d, nn.BatchNorm2d}, dtype=torch.qint8), [['weight', 'bias']])
graph = torch.fx.symbolic_trace(bn)

# Graph begins
graph_str = str(graph)

# Inputs to the model
x = torch.randn(1, 4, 4, 4)