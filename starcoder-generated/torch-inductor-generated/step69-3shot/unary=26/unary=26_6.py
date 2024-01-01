
class LeakyReLU(torch.nn.Module):
    def __init__(self, channel_num, negative_slope):
        super().__init__()
        self.channel_num = channel_num
        self.conv_t = torch.nn.ConvTranspose2d(channel_num, 64, 3,
                                             stride=2, padding=1,
                                             bias=True)
        self.negative_slope = negative_slope
        self.output = torch.nn.Linear(32*2*2, 3)
    def forward(self, x):
        x = torch.tanh(self.conv_t(x))
        x = leaky_relu_(x, self.negative_slope)
        x = x > 0
        x = x.view(x.shape[0],-1)
        return self.output(x)
