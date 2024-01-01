
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.ConvTranspose2d(64, 128, kernel_size=3)
        self.conv1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=3)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = self.conv1(v2)
        v3 = torch.relu(v3)  
        return v3
# Inputs to the model
x1 = torch.randn(1, 64, 100, 100)
