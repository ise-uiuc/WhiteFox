
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.a = 0
        self.b = 0

    def forward(self, x1):
        if self.a > 0:
            if self.b > 0:
                v1 = torch.relu6(self.conv1(x1))
            else:
                v1 = x1 * 0
        else:
            if self.b > 0:
                v1 = 6 + torch.relu6(self.conv1(x1))
            else:
                v1 = self.a + self.b

        v2 = torch.relu6(self.a + self.b)

        if self.a > torch.relu6(v2):
            v3 = torch.relu6(v2) + 3
        else:
            v3 = torch.relu6(v2) * 2

        v4 = torch.sigmoid(v3 + v2)

        v5 = torch.relu6(v2 + v2) * v4

        v6 = v5 + 3

        v7 = torch.relu6(v6)

        if v7 < 0:
            x1 = torch.relu6(v7)
        else:
            x1 = 0
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
