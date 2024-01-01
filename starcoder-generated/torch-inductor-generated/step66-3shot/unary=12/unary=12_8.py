
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 5, 3, stride=2, padding=0, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(5, 5, 3, stride=2, padding=0, dilation=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
model2 = Model2()
model2.load_state_dict(torch.load('model.pth'))
model2.eval()
# Inputs to the model
x1 = torch.Tensor(1, 3, 64, 64)
