
class C1(nn.Module):
    def __init__(self):
        super(C1, self).__init__()
        self.conv1d_1 = nn.ConvTranspose1d(input_channels=1,
                                         output_channels=12,
                                         kernel_size=[1],
                                         stride=1,
                                         padding=0,
                                         bias=False)
    def forward(self, x):
        y = self.conv1d_1(x)
        z = y > 0
        return z
# Inputs to the model
x13 = torch.randn(5, 1, 4)
