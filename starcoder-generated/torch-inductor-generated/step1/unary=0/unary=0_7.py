
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
 
    def forward(self, x):
        _convolution_1 = self.conv(x) # 0.044715 * _convolution_1 + _convolution_1  # 0.7978845608028654 * torch.tanh((_convolution_1 + _convolution_1) + 1) # _convolution_1 * _convolution_1 * _convolution_1
        v1 = _convolution_1 * _convolution_1 * _convolution_1
        v3 = v1 * 0.044715
        v4 = v1 + 0.044715 
        v5 = v4 + _convolution_1
        v6 = v5 * 0.7978845608028654
        v7 = torch.tanh(v6)
        v8 = v3 + v6
        v9 = v8 + 1
        v11 = _convolution_1 + v9
        return v9

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
