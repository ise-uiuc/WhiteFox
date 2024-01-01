
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=False)
        v2 = F.interpolate(x2, size=x1.size()[2:], mode='bilinear', align_corners=False)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 100, 100)
x2 = torch.randn(1, 1, 200, 200)
