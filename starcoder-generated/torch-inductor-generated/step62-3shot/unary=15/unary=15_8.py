
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in0 = torch.nn.Conv2d(256, 256, 1, stride=1, padding=0)
        self.in1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.out0 = torch.nn.ReLU()
        self.out1 = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.in0(x1)
        v2 = self.in1(v1)
        v3 = self.out0(v2)
        v4 = self.out1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 256, 55, 55)
