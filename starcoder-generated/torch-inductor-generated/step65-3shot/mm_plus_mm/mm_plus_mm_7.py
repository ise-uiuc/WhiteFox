
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_1=1, padding_2=1):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=padding_1),  # dim x dim kernel, padding = 'padding_1'
                                        nn.BatchNorm2d(dim),
                                        nn.ReLU(True),
                                        nn.Conv2d(dim, dim, 3, padding=padding_2),  # dim x dim kernel, padding = 'padding_2'
                                        nn.BatchNorm2d(dim))
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
# Inputs to the model
input_tensor = torch.randn(2, 32, 16, 16)
