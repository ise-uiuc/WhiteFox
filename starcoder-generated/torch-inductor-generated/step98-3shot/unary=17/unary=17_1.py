
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=3)
        self.conv_t2 = torch.nn.ConvTranspose2d(1, 2, kernel_size=3, stride=4)
    def forward(self, x1):
        v1 = self.conv_t1(x1)
        v2 = self.conv_t2(v1)
        v3 = torch.relu(v2)
        v4 = v3.transpose(1, 2)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
