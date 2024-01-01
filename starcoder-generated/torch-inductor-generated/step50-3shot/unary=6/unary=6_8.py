
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(2, stride=2)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        t1 = self.maxpool(x1)
        t2 = self.tanh(t1)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 20, 20)
