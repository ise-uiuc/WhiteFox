
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(10, 16, 3) 
        self.conv2 = torch.nn.ConvTranspose2d(16, 8, 4) 
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.stride(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 10, 16, 16)
