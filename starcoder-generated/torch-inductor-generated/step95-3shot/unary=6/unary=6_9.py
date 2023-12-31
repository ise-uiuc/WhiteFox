
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.quant0 = torch.quantization.QuantStub()
        self.quant1 = torch.quantization.QuantStub()
        self.conv0 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.bias0 = torch.nn.Parameter(torch.zeros(3, 3, 5, 5))
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(3, 3, 5, stride=1, padding=2)
        self.bias1 = torch.nn.Parameter(torch.zeros(3, 3, 5, 5))
        self.sigmoid = torch.nn.Sigmoid()
        self.add = torch.nn.quantized.FloatFunctional()
        self.quant2 = torch.quantization.DeQuantStub()
        self.quant3 = torch.quantization.DeQuantStub()
    def forward(self, x0, x1):
        v0 = self.quant0(x0)
        v1 = self.quant1(x1)
        v2 = self.conv0(v1)
        v2 = torch.quantize_per_tensor(v2, 0.05153478597640991, 0, torch.quint8)
        v2 = torch.add(v2, self.bias0)
        v2 = self.sigmoid(v2)
        v3 = self.relu(v2)
        v4 = self.conv1(v3)
        v4 = torch.quantize_per_tensor(v4, 0.012193959850027084, 0, torch.quint8)
        v4 = torch.add(v4, self.bias1)
        v4 = self.sigmoid(v4)
        v5 = self.add.add_relu(v0, v4)
        v6 = self.quant2(v1)
        v7 = self.quant3(v5)
        return v7
# Inputs to the model
x0 = torch.randn(1, 3, 32, 32)
x1 = torch.randn(1, 3, 32, 32)
