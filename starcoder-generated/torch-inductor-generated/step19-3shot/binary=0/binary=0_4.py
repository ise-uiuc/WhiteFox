
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(7, 5, 1, stride=1, padding=1)
    def forward(self, x1, x2, other=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(x2)
        if other is None:
            v3 = v1
        elif v2.shape[0] < 8:
            if v1.shape[0] == v2.shape[0]:
                if v2.shape[0] == 2:
                    v3 = torch.rand(v1.shape)
                else:
                    v3 = v1
        if v3.shape[0] == 2:
            v3 = v3 + v2
        elif v2.shape[1] < 8:
            if v1.shape[1] == v2.shape[1]:
                v2 = v2 + v1
        if v3.shape[1] == 5:
            v2 = v1 + v2
            v3 = v2 + v3
        elif v3.shape[0] == 3:
                v3 = torch.rand(v3.shape)
        elif v3.shape[1] == 1:
            if v2.shape[1] == 5:
                v3 = v2 + v3
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 7, 64, 64)
