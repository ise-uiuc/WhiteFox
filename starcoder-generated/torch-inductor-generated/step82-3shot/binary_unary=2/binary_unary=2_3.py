
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2,2,2, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(1,1,3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        t1 = v1 - 1
        v4 = self.conv2(t1)
        t2 = v4 - 2
        v7 = t2
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 5, 5)
