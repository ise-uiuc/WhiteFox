
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 8 groups with 24 channels for a total of 3,12
        self.conv2 = nn.Conv2d(24, 3, kernel_size=1, groups=8)
        # batchnorm2d layer groups is set to 8
        self.bn = nn.BatchNorm2d(3)

    def forward(self, y):
        # The 8 groups are each responsible for 4
        # channels, so 3, 8, 3 becomes 3, 24

        # Expand (24 to 3, 8, 1, 1)
        a = F.unfold(y, kernel_size=4, 
                    padding=0, stride=4).view(1, 24, 3, 1, 1)
        # b[0] = 8 (groups), b[1] = 4 (input channels), b[2] = 1 (groups), b[3] = 3 (output channels), b[4] = 1 (output channels)
        b, c, d, e, f = a.size()
        # Flatten it to (24, 16). This becomes a shape of 1, 24, 16 (equivalent to 16, 24)
        c = torch.flatten(a, start_dim=b, end_dim=c)
        # Reshape to (1, 24, 8).
        c = c.view(d, e, f)
        # Convolve by calling conv2d (output is 1, 8, 16)
        d = self.conv2(c)
        # Reshape to (1, 24, 16).
        d = d.flatten(start_dim=0, end_dim=1)

        # Expand (16, 24 to 1, 1, 16, 24)
        a = F.unfold(d, kernel_size=4, 
                    padding=0, stride=4).view(1, 3, 8, 1, 16)
        # b[0] = 8 (groups), b[1] = 4 (input channels), b[2] = 16 (groups), b[3] = 3 (output channels), b[4] = 24 (output channels)
        b, c, d, e, f = a.size()
        # Flatten it to (24, 64). This becomes a shape of 1, 8, 32 (equivalent to 16, 24)
        c = torch.flatten(a, start_dim=b, end_dim=c)
        # Reshape to (1, 8, 8).
        c = c.view(d, e, f)
        # Batchnorm (8)
        e = self.bn(c)
        # Unfold

        # Returns (1, 24, 16)
        return e

torch.manual_seed(1)
a1 = torch.randn(1, 512) 
a2 = torch.randn(1, 256)
b = torch.randn(1)

d = torch.cat((a1, a2), 1)
