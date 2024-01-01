
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(450, 366, 3, stride=2, padding=0)
    def forward(self, x):
        l1 = self.conv_t(x)
        l2 = l1 > 0
        l3 = l1 * -0.5586
        l4 = torch.where(l2, l1, l3)
        return torch.nn.functional.interpolate(l4, size=17, mode='bicubic', align_corners=False)
# Inputs to the model
x = torch.randn(978, 450, dtype=torch.float, device='cuda')
