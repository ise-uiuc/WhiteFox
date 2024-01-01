
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = []
        for i in range(2):
            self.conv1.append(torch.nn.Conv2d(5, 5, 5, stride=1, padding=5))
        self.conv2 = []
        for i in range(2):
            self.conv2.append(torch.nn.Conv2d(5, 5, 5, stride=1, padding=6))
    def forward(self, x1, other=False):
        v1 = self.conv1[0](x1)
        if other == False:
            other = torch.randn(v1.shape)
        v2 = self.conv2[0](v1) + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 5, 64, 64)
