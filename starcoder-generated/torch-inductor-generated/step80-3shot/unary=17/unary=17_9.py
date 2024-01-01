
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 2, stride=1, padding=int(1), bias=False)
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=int(0), bias=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        v5 = torch.conv1d(v4)
        ret = v5.transpose(2, 1)
        v6 = torch.max(ret,dim=1)
        v7 = torch.mm(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 1)
