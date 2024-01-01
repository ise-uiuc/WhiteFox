
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = v1.to(memory_format=torch.channels_last)
        v3 = v2.unsqueeze(1)
        v4 = v3.squeeze(2)
        v5 = v4.permute(2, 1, 0)
        v6 = v5.contiguous()
        return v1, v4, v6
x1 = torch.randn([256, 3, 64, 64])
x2 = torch.randn([256, 64, 32, 32])
