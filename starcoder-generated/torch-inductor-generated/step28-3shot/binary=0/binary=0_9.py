
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 1, stride=1, padding=0)
    def forward(self, x1, padding1=None):
        v1 = self.conv(x1)
        if padding1 == None:
            padding1 = torch.randn(v1.shape)
        v4 = v1.mean(dim=1).unsqueeze(1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
