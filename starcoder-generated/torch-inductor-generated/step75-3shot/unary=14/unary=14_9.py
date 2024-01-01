
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(4,3,2,2,0,False)
        self.conv2 = torch.nn.ConvTranspose2d(3,4,2,2,0,False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = v1.sigmoid()
        v1 = v1*1
        v2 = self.conv2(v1)
        v3 = v2.tanh()
        v4 = v2 * 0
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
