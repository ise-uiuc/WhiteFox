
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prelu1 = torch.nn.PReLU(num_parameters=1, init=0.25)
        self.prelu2 = torch.nn.PReLU(num_parameters=1, init=0.5)
    def forward(self, x1, value):
        v1 = self.prelu1(x1)
        v2 = self.prelu2(v1 + value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
other = 1
