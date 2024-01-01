
class Model(torch.nn.Module):
    def __init__(self, input_channel=3, input_dim=(1, 7), output_dim=(3, 1), kernel_size=16):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_dim[0]
        self.kernel_size = (1, 1)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.maxpool = torch.nn.MaxPool2d((1, 3), (1, 3))
        self.conv_t = torch.nn.ConvTranspose2d(self.output_channel, self.output_channel, self.kernel_size,
                                    stride=(3, 3))
        self.pad1 = torch.nn.ReplicationPad3d((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0))
        self.pad2 = torch.nn.ReplicationPad2d((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.pad_w = torch.nn.ReplicationPad2d((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.pad_h = torch.nn.ReplicationPad2d((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
        self.upsample = torch.nn.Upsample(2, 'bilinear')
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.upsample(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 100, 32)
