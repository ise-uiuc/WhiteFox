
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(25, 5, 1, stride=1, padding=1)
    def forward(self, x1, other=0.1, padding1=None, padding2=None, padding3=None, padding4=None, other1=0.1):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        if padding2 == None:
            padding2 = torch.randn(v1.shape)
        if padding3 == None:
            padding3 = torch.randn(v1.shape)
        if padding4 == None:
            padding4 = torch.randn(v1.shape)
        v2 = v1 + 0.1
        v3 = v1 + 0.1
        v4 = v2 + 0.1
        v5 = v3 + 0.1
        v6 = v4 + 0.1
        v7 = v5 + other
        v8 = v6 + 0.1
        v9 = v7 + other1
        return v9
# Inputs to the model
x1 = torch.randn(1, 25, 100)
