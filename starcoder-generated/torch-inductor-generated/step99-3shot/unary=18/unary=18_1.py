
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 10, (5, 5), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(10, 5, (5, 5), stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(5, 7, (5, 5), stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(7, 8, (5, 5), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.nn.functional.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.nn.functional.tanh(v5)
        v7 = self.conv4(v6)
        v8 = torch.nn.functional.tanh(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 6, 28, 28)
