
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tcd = torch.nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        self.tcd_relu = torch.nn.ReLU()
        self.tcd_bn = torch.nn.BatchNorm2d(32)

    def forward(self, x1, x2, x3, x4, x5, x6, x7, x8_1, x8_2):
        x8_1_out = self.tcd(x8_1)
        x8_1_out = self.tcd_relu(x8_1_out)
        x8_1_out = self.tcd_bn(x8_1_out)
        x8_2_out = self.tcd(x8_2)
        x8_2_out = self.tcd_relu(x8_2_out)
        x8_2_out = self.tcd_bn(x8_2_out)
        x9 = torch.cat([self, x8_1_out, x8_2_out], dim=1)
        x10 = x9[:, 0:1792]
        x11 = x10[:,0:1792]
        x12 = torch.cat([x9, x11], dim=1)
        x13 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        x14 = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False)
        x15 = torch.nn.ReLU()
        x16 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False)
        x17 = torch.nn.ReLU()
        x18 = torch.nn.BatchNorm2d(16)
        x19 = torch.nn.Hardtanh(min_val=0.0, max_val=6.0, inplace=False)
        x20 = torch.nn.BatchNorm2d(8)
        x21 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        x22 = torch.nn.ReLU()
        x23 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        x24 = torch.nn.ReLU()
        x25 = torch.nn.BatchNorm2d(64)
        x26 = torch.cat([x23, x25], dim=1)
        x27 = self.tcd0_relu(x26)
        x28 = self.tcd0_bn(x27)
        x29 = self.tcd1_relu(x28)
        x30 = self.tcd1_bn(x29)
        x31 = torch.cat([x30, x23], dim=1)
        x32 = self.tcd2_relu(x31)
        x33 = self.tcd2_bn(x32)
        x34 = torch.cat([x33, self, x8_1_out, x8_2_out], dim=1)
        x35 = x34[:, 0:131072]
        x36 = x35[:, 0:16384]
        x37 = torch.cat([x34, x36], dim=1)
        x38 = torch.nn.LSTM(input_size=3, hidden_size=32, num_layers=None, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
        x39 = torch.nn.Dropout(p=0.5, inplace=False)
        x40 = self.tcd3_relu(x37)
        x41 = self.tcd3_bn(x40)
        x42 = torch.cat([x41, x41], dim=1)
        x43 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)
        x44 = self.tcd4_relu(x43)
        x45 = self.tcd4_bn(x44)
        x46 = torch.cat([x45], dim=1)
        x47 = x16(x46)
        output = x18(x47)
        return output

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 32, 28, 28)
x2 = torch.randn(1, 32, 14, 14)
x3 = torch.randn(1, 32, 7, 7)
x4 = torch.randn(1, 32, 28, 28)
x5 = torch.randn(1, 32, 14, 14)
x6 = torch.randn(1, 32, 7, 7)
x7 = torch.randn(1, 32, 28, 28)
x8_1 = torch.randn(1, 3, 64, 64)
x8_2 = torch.randn(1, 3, 64, 64)
