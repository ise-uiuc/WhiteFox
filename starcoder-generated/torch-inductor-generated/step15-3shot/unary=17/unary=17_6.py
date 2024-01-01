
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3,16,3)
        self.conv_2 = torch.nn.ConvTranspose2d(16,32,3)
        self.conv_3 = torch.nn.Conv2d(32,32,3)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1,3,224,224)
