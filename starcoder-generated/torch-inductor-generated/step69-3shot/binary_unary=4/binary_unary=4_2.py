
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.quantized.Linear(3, 6, dtype=torch.qint8)
 
    def forward(self, x2, other=0):
        return self.linear(x2) + other

# Initializing the model
m = Model()
m.qconfig = torch.quantization.QConfig(
    activation=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8),
    weight=torch.quantization.PlaceholderObserver.with_args(dtype=torch.qint8))
 
class PlaceholderModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return torch.ops.quantized.placeholder(x, dtype=self.dtype)

_ = PlaceholderModule(dtype=torch.qint8) # Making sure PlaceholderModule will use qint8

# Inputs to the model
x2 = torch.randn(10, 3, dtype=torch.float32)
x7 = torch.randn(10, 3, dtype=torch.float32)
