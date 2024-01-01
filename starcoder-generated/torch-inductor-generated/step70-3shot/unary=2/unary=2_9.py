
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool2d = torch.nn.MaxPool2d(4, stride=4, return_indices=True)
        self.mul = torch.nn.Mul()
        self.tanh = torch.nn.Tanh()
        self.hardtanh = torch.nn.Hardtanh()
        self.batchnorm2d = torch.nn.BatchNorm2d(5, 1, 1)
        self.hardsigmoid = torch.nn.Hardsigmoid()
        self.hardswish = torch.nn.Hardswish()
        self.softplus = torch.nn.Softplus()
        self.softmax = torch.nn.Softmax()
        self.leaky_relu = torch.nn.LeakyReLU()
    def forward(self, x1):
        v1, v3_return = self.maxpool2d(x1)
        v2 = self.mul(v1, 0.5)
        v4 = self.mul(v3_return.clone(), v3_return)
        v5 = self.tanh(v2)
        v6 = self.hardtanh(v5)
        v7 = self.batchnorm2d(v4)
        v8 = self.hardsigmoid(v7)
        v9 = self.hardswish(v8)
        v10 = self.softplus(v9)
        v11 = self.softmax(v10)
        v12 = self.leaky_relu(v11)
        return v12
# Inputs to the model
x1 = torch.randn(1, 5, 16, 16)
