
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, groups=1)
    def forward(self, x1, x2, x3):
        x = self.conv(x1)
        x1 = x + x2
        x1 = self.conv(x1)
        x = self.conv(torch.cat((x1, x2), dim=1))
        x1 = torch.cat((x1, x3), dim=1)
        x2 = x1 + x
        x = self.conv(torch.cat((x1, x2)))
        x1 = x + self.conv(torch.cat((x1, x2)))
        x1 = x + self.conv(x)
        x2 = self.conv(x) + self.conv(x)
        x2 = x2 + x1
        x = self.conv(x1 + x2) + torch.cat((x1, x2))
        x = self.conv(x1 + self.conv(x2)) + torch.cat((x1, x2))
        x1 = torch.cat((x1, x2)) + self.conv(x)
        x1 = x + torch.cat((x1, x2))
        x1 = x + torch.cat((x1, x2), dim=1)
        x1 = torch.cat((x1, x2), dim=1) + torch.cat((x1, x2), dim=1)
        x = x1 + torch.cat((x1, x2), dim=1)
        x = x + x1
        x = torch.cat([torch.cat((x1, x2), dim=1), torch.cat((x1, x2), dim=1)])
        x1 = torch.cat((x, x1), dim=1) + x
        x = x1 + x
        x1 = x + x
        x2 = torch.cat((x, x1, x), dim=1)
        x = self.conv(torch.cat((x1, x2), 1))
        x = x + torch.cat((x1, x2), dim=1)
        return x
# Inputs to the model
x1 = torch.randn(3, 1, 8, 8)
x2 = torch.randn(3, 1, 8, 8)
x3 = torch.randn(3, 1, 8, 8)
