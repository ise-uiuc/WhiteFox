
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(32, 32, 5, padding=0)
        self.conv2 = torch.nn.ConvTranspose2d(32, 1, 2, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.max_pool(v3,kernel_size=v3.size()[2:]).view(v3.size()[0],-1)
        return v4
# Inputs to the model
x1 = torch.randn(1, 32, 30, 30)
