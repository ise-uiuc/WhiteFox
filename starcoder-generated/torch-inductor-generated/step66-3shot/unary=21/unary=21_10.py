
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(32,8,5,1,2)
    def forward(self, x):
        n=self.conv(x)
        return n
# Inputs to the model
x = torch.randn(1, 32, 256, 256)
