
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(3,3,5)
    def forward(self,x):
        x = self.conv(x)
        x = torch.nn.functional.relu(x)
        return x
# Input shape
x = torch.randn(1,3,5,5)
