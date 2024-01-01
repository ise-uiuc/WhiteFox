
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conva = torch.nn.Conv2d(32, 96, 1, stride=1, padding=0)
        self.convb = torch.nn.Conv2d(96, 128, 1, stride=1, padding=0)
    def forward(self, x1, x2=torch.randn(1,1,0,0), x3=None):
        v1 = self.conva(x1)
        v2 = v1 + 3
        if not x3:
            x3 = (v2 + x2 ).view(list(v2.shape)+[-1]).contiguous()    # This reshapes x2 from N*C*H*W --> N*C*H*W*1
        v3 = self.convb(x3)
        v4 = v3 + 2
        return v4
# Inputs to the model
x1 = torch.randn(2, 32, 4, 4)
