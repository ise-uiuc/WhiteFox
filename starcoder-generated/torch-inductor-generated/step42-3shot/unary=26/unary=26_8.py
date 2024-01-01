
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 9, stride=4, padding=0, bias=False)
    def forward(self, x2):
        r1 = self.conv_t(x2)
        r2 = r1 > 0
        r3 = r1 * -0.4
        r4 = self.sub_850(torch.tensor([1.0], device=r1.device), r1) # Subnode
        r5 = torch.where(r2, r1, r3)
        r6 = r5 + r4
        return r6
# Subgraph begins
class SubModule_1206(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x6):
        s1 = x6 + torch.tensor([0.0485], device=x6.device)
        s2 = x6 + torch.tensor([0.2417], device=x6.device)
        s3 = x6 + torch.tensor([0.1407], device=x6.device)
        s6 = x6 * torch.tensor([0.1], device=x6.device)
        return s6
# Subgraph ends
def forward(self, x2):
    r1 = self.conv_t(x2)
    r2 = r1 > 0
    r3 = r1 * -0.4
    r4 = SubModule_1206()(r1)
    r5 = self.add_695(r1, r4)
    r6 = torch.where(r2, r1, r3)
    r7 = r1 + r4
    return r7
# Inputs to the model
x2 = torch.randn(72, 1, 96, 96)
