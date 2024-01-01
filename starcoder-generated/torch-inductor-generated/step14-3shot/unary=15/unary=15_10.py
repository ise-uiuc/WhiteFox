
class Model(torch.nn.Module):
    def __init__(self, in_channel=64, kernel_size=3):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=64, kernel_size=kernel_size, padding=int(kernel_size % 2!= 0))
        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=int(kernel_size % 2!= 0))
        self.conv2_bn = torch.nn.BatchNorm2d(128)
        self.conv2_act = torch.nn.PReLU(128)
        self.conv3 = torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=kernel_size, padding=int(kernel_size % 2!= 0))
        self.conv3_act = torch.nn.PReLU(64)
    def forward(self, x, x_len):
        v0 = x
        if x_len is not None:
            v0 = torch.nn.utils.rnn.pack_padded_sequence(v0, x_len.cpu(), batch_first=False)
        v1 = self.conv1_bn(self.conv1(v0))
        v2 = self.conv2_act(self.conv2_bn(self.conv2(v1)))
        v3 = self.conv3_act(self.conv3(v2))
        if x_len is not None:
            v3 = torch.nn.utils.rnn.pad_packed_sequence(v3, batch_first=False)[0]
        out_len = torch.nn.functional.max_pool2d(torch.nn.functional.interpolate(x_len.unsqueeze(1).float().view(1, 1, x_len.shape[0], 1), scale_factor = 2), kernel_size = v3.shape[-1]).squeeze().long()
        return v3, out_len
# Inputs to the model
x = torch.randn(5, 32, 334, 8)
x_len = torch.tensor([330, 328, 318, 310, 292])
