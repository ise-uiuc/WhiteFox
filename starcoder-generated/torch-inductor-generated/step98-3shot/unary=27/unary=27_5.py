
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(160, 7, (1, 7), stride=(1, 1), padding=(1, 7), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.conv26 = torch.nn.Conv2d(6, 14, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.maxpool = torch.nn.MaxPool2d((1, 4), stride=(1, 4), padding=0, dilation=1, return_indices=True, ceil_mode=False)
        self.convt = torch.nn.ConvTranspose2d(1, 14, (1, 4), stride=(1, 2), padding=(0, 1), dilation=(1, 1), groups=1, bias=True, padding_mode='zeros')
        self.min = min
        self.max = max
    def forward(self, x):
        v0 = x.size()
        v2 = self.conv1(x)
        v3 = self.conv26(v2)
        v4, v5 = self.maxpool(v3)
        v6 = self.convt(v4)
        v7 = v6.view(-1, v0[1], v0[2])
        v8 = x.view(-1, v7.size()[2], v6.size()[3], v6.size()[4])
        v9 = torch.matmul(v8, v7)
        v10 = v9.matmul(v9)
        v11 = torch.clamp_min(v10, self.min)
        v12 = torch.clamp_max(v11, self.max)
        v13 = v12 * v12
        v14 = v13.view(x.size())
        return v14
min = 0.39
max = 0.04
# Inputs to the model
x = torch.randn(2, 1, 30, 40)
