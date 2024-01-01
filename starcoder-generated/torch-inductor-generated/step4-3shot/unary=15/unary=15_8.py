
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_bn_relu = nn.Sequential(
            nn.Conv3d(3, 8, 7, stride=(1, 2, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_bn_relu(x)
        return x
# Inputs to the model
x1 = torch.randn(1, 3, 32, 64, 128)
