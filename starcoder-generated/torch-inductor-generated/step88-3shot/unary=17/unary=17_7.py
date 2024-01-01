
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = nn.Conv3d(3, 32, kernel_size=3, stride=(2, 2))
        self.conv_2 = nn.Conv3d(32, 32, kernel_size=3, stride=(2, 2))
        self.conv_3 = nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2))
    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 4, 32, 32)
