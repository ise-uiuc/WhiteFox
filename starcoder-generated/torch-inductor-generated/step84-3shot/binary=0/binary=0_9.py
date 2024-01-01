
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 16, 1, stride=1, padding=0)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x1, padding1=False, padding2=True):
        v1 = self.conv(x1)
        if padding1 == False and padding2 == True:
            v1 = self.dropout(v1)
        return v1
# Inputs
x1 = torch.randn((1, 64, 80, 80), device='cpu')
