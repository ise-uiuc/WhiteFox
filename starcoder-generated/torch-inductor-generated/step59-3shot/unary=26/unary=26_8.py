
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 16, 3, stride=1, bias=False)
        self.fc2 = torch.nn.Linear(16, 16, bias=False)
        self.conv_t4 = torch.nn.ConvTranspose2d(32, 480, 1)
        self.conv_t5 = torch.nn.ConvTranspose2d(20, 140, 1, stride=1, bias=False)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(20, 16, 1, bias=False)
        self.fc6 = torch.nn.Linear(32, 48, bias=False)
        self.fc64 = torch.nn.Linear(48, 282, bias=False)
        self.relu0 = torch.nn.ReLU()
        self.conv_t7 = torch.nn.ConvTranspose2d(2, 64, 3, stride=2, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.conv_t9 = torch.nn.ConvTranspose2d(80, 128, 1)
        self.fc12 = torch.nn.Linear(160, 64, bias=False)
        self.conv_transpose000 = torch.nn.ConvTranspose2d(80, 1, 8, padding=0, output_padding=(1, 1), bias=False)
    def forward(self, input_x):
        x0 = torch.nn.functional.gelu(self.conv_t(input_x))
        x1 = self.relu0(x0)
        x2 = torch.nn.functional.gelu(self.conv_t4(x1))
        x3 = torch.nn.functional.gelu(self.conv_t5(x2))
        x4 = self.conv_transpose3(x3)
        x5 = torch.nn.functional.gelu(self.fc6(x4))
        x6 = torch.nn.functional.gelu(self.fc64(x5))
        x7 = x6
        x8 = x6 > 0
        x9 = x6 * -1
        x10 = torch.where(x8, x6, x9)
        x11 = x10
        x12 = torch.matmul(x11, x7.transpose(-1, -2))
        x13 = x12 > 0
        x14 = x12 * -1
        x15 = torch.where(x13, x12, x14)
        x16 = x15
        x17 = torch.matmul(x16, x7)
        x18 = torch.nn.functional.tanh(x17)
        x19 = x18
        x20 = x18 > 0
        x21 = x18 * -1
        x22 = torch.where(x20, x18, x21)
        x23 = torch.matmul(x22, x7) + torch.matmul(x11, self.fc2.weight.transpose(-1, -2))
        x24 = x23
        x25 = torch.nn.functional.adaptive_avg_pool2d(x24, (1, 1))
        x26 = x25 > 0
        x27 = x25 * -1
        x28 = torch.where(x26, x25, x27)
        x29 = torch.matmul(x23, x28)
        x30 = x29 > 0
        x31 = x29 * -0.08
        x32 = torch.where(x30, x29, x31)
        x33 = self.relu2(x32)
        x34 = x33
        x35 = x33 * 0.08
        x36 = torch.nn.functional.gelu(self.conv_t7(x34))
        x37 = x36
        x38 = torch.nn.functional.gelu(self.conv_t9(x37))
        x39 = torch.nn.functional.gelu(self.fc12(x36))
        x40 = x39 > 0
        x41 = x39 * -1
        x42 = torch.where(x40, x39, x41)
        x43 = x35
        x44 = torch.matmul(x42, x43.transpose(-1, -2))
        x45 = x44 > 0
        x46 = x44 * -1
        x47 = torch.where(x45, x44, x46)
        x48 = x47
        x49 = torch.matmul(x48, x49)
        x50 = x49 > 0
        x51 = x49 * -1
        x52 = torch.where(x50, x49, x51)
        x53 = torch.matmul(x44, x50.transpose(-1, -2))
        x54 = torch.nn.functional.sigmoid(x53)
        x55 = x54
        x56 = torch.matmul(x53, self.fc64.weight)
        x57 = x52
        x58 = torch.matmul(x55, self.conv_transpose3.weight)
        x59 = torch.cat((x56, x57, x57), dim=1)
        x60 = x59
        x61 = torch.nn.functional.adaptive_avg_pool2d(x60, 1)
        x62 = self.relu0(x61)
        x63 = self.relu0(x62)
        x64 = x63
        x65 = x63 * 0.16
        x66 = torch.nn.functional.gelu(self.conv_t7(x64))
        x67 = x66
        x68 = x65
        x69 = torch.cat((x67, x68), dim=1)
        x70 = torch.nn.functional.gelu(self.fc12(x69))
        x71 = x70
        x72 = x70 * 0.125
        x73 = torch.cat((x71, x72), dim=1)
        x74 = x73
        x75 = torch.nn.functional.gelu(self.conv_transpose000(x74))
        return x75
# Inputs to the model
input_x = torch.randn(8, 16, 83, 27)
