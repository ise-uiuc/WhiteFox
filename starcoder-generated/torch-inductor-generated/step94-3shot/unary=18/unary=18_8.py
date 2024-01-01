
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 1, (14, 11), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 1, (14, 11), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        if torch.bernoulli(0.1) > 0:
            return v1
        else:
            return self.conv2(v1)
# Inputs to the model
x1 = torch.randn(1, 4, 33, 108)
