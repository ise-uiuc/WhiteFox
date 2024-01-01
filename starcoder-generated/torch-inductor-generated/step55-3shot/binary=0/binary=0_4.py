
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 8, 3, stride=1, padding=1)
    def forward(self, x1, other1, other2, other3, padding1=None, padding2=None):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 + other1
        v4 = v3 + other2
        if padding1 == None:
            padding1 = torch.randn(v3.shape)
        if padding2 == None:
            padding2 = torch.randn(v4.shape)
        v5 = v4 + padding1
        v6 = v4 + v5 + other3
        v7 = v6 + other3
        v8 = v7 + other3
        return v8
# Inputs to the model
x1 = torch.randn(12, 3, 128, 128)
other1 = torch.randn(12, 8, 128, 128)
other2 = torch.randn(12, 8, 128, 128)
other3 = torch.randn(12, 8, 128, 128)
