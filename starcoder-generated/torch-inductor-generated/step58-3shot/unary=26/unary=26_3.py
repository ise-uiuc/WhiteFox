
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, out_channels=9, kernel_size=(7, 6), stride=2, bias=True, dilation=1, padding=1)
        self.dropout = torch.nn.Dropout2d(inplace=False)
        self.relu = torch.nn.ReLU()
        self.conv1d_0 = torch.nn.Conv2d(2, in_channels=4, out_channels=47, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.sigmoid_0 = torch.nn.Sigmoid()
        self.hardtanh_0 = torch.nn.Hardtanh(0.0011904610711298627, 0.6236021280772245)
    def forward(self, x11):
        x1 = self.conv_t(x11)
        x2 = self.dropout(x1)
        x3 = self.relu(x2)
        x4 = self.conv1d_0(x1)
        x5 = self.sigmoid_0(x4)
        x6 = self.hardtanh_0(x5)
        x7 = x6 > 0.0
        x8 = x6 * 0.31
        x9 = torch.where(x7, x6, x8)
        x10 = torch.cat([x3, x9], 1)
        return x10
# Inputs to the model
x11 = torch.randn(1, 3, 4, 3, 2)
