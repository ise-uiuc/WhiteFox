
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = torch.nn.Conv2d(4, 32, (5, 33), stride=(2, 15), padding=(1, 7), dilation=(2, 8), groups=16, bias=True)
        self.model2 = torch.nn.Conv2d(4, 32, (5, 33), stride=(2, 15), padding=(1, 7), dilation=(2, 8), groups=16, bias=True)
        self.model3 = torch.nn.Conv2d(4, 32, (5, 33), stride=(2, 15), padding=(1, 7), dilation=(2, 8), groups=16, bias=True)
    def forward(self, x1, x2, x3):
        v1 = self.model1(x1)
        v2 = self.model2(x2)
        v3 = self.model3(x3)
        v4 = v1 + v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 4, 1024, 416)
x2 = torch.randn(1, 4, 1024, 416)
x3 = torch.randn(1, 4, 1024, 416)
