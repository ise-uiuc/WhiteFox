
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.bn = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        x = F.conv2d(input=x, weight=F.conv_transpose2d(input=x,
                                                            weight=F.conv_transpose2d(input=torch.transpose(x, 1, 2),
                                                                                      weight=torch.transpose(self.conv(x).transpose(1, 2)).transpose(2, 3),
                                                                                      stride=(5, 20)),
                                                            stride=(20, 5)))
        x = F.relu(self.bn(x))
        return x
# Inputs to the model
x = torch.randn(1, 1, 300, 500)
