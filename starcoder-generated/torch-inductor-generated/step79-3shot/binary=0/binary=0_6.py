
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1, padding=1)
        self.layer = torch.nn.Linear(128, 128)
    def forward(self, x1, padding1=None, padding2=True):
        v1 = self.conv(x1)
        if padding2 == True and padding1 == None:
            padding1 = torch.randn(v1.shape)
        v2 = self.layer(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 28, 28)
