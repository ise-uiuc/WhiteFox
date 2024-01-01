
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(in_channels= 3, out_channels= 64, kernel_size= 11, stride= 4, padding= 2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = v1 - 31
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
