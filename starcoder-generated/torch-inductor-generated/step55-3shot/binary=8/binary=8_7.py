
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 8, 1, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(8)
        self.bn2 = torch.nn.BatchNorm2d(8)
        self.bn3 = torch.nn.BatchNorm2d(8)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(x)
        v3 = v1 + v2
        v4 = self.bn1(v3)
        v5 = self.bn2(v3)
        v6 = self.bn3(v3)
        p1 = v4.permute(0, 3, 1, 2)
        p2 = v6.permute(0, 3, 1, 2)
        p3 = torch.cat((p1, p2), dim=3)
        p4 = p3.flatten(1)
        l1 = p4.size()[0]
        p5 = p4.reshape(l1, 2, 16)
        l2 = p5.size()[0]
        p6 = p5.reshape(l2, 16, 2)
        c1 = torch.cholesky(p6)
        l3 = c1.size()[0]
        c2 = c1.reshape(l3, 2, 2, 4)
        (p1, p2, p3, g1, g2) = c2.contiguous().split([2, 2, 2, 4], dim=1)
        g1 = g1.squeeze(1)
        g2 = g2.squeeze(1)
        g3 = [g1, g2]
        g4 = torch.cat(g3, dim=-1)
        x1 = g4.unsqueeze(0)
        p1 = p1.permute(0, 1, 3, 2)
        p2 = p2.permute(0, 1, 3, 2)
        g5 = g4.permute(0, 2, 1)
        g6 = g5.unsqueeze(0).expand_as(x)
        s1 = torch.bmm(x, p2)
        s2 = torch.bmm(x1, g4)
        s3 = torch.bmm(s1, g5).squeeze(0).permute(2, 0, 1)
        s4 = torch.bmm(s2, g6).squeeze(0)
        s5 = s3 * s4
        g7 = s5.squeeze(1).reshape(1, 1, *x.size())
        bn2d = self.bn3(g7)
        soft = self.softmax(bn2d)
        return (g4, soft)
# Inputs to the model
x = torch.randn(16, 1, 32, 32)
