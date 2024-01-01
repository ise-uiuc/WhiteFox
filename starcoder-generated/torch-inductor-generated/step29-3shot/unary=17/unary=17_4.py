
class Model(nn.Module):
    def __init__(self, model_op, kernel_op, pad_op, bias_op, stride_op, in_channels_op, out_channels_op, kernel_size_op):
        super(Model, self).__init__()
        self.kernel_op = kernel_op
        self.pad_op = pad_op
        self.stride_op = stride_op
        self.out_channels_op = out_channels_op
        self.conv = nn.ConvTranspose2d(in_channels_op, out_channels_op, kernel_size_op, (stride_op, 4),
                                       (1, pad_op), bias_op)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2048, 7, 7)
