
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.zero_point_linear1 = torch.nn.quantized.Linear(192, 208, bias=True)
        self.prelu1 = torch.nn.PReLU(208)
        self.zero_point_linear2 = torch.nn.quantized.Linear(208, 96, bias=True)
    def forward(self, x1):
        v1 = self.zero_point_linear1(x1)
        v2 = self.prelu1(v1.float())
        v3 = v2.to(torch.quint8)
        v4 = self.zero_point_linear2(v3)
        return v4
# Inputs to the model
x1 = torch.randn(28, 192)
