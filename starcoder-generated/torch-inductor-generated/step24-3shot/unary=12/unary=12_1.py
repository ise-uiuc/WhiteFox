
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 2, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        vt = v1.sigmoid()
        v3 = torch._C._nn.ctc_loss(v1, vt.cpu()) * v1
        return v3
# Inputs to the model
v1 = torch.randn(1, 128, 8)
v2 = v1.sigmoid()
v3 = torch.nn.functional.ctc_loss(v1, v2, (torch.arange(v1.size(1)).long(), torch.arange(v1.size(1)).long()), 2)
