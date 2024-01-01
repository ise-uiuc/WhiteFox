
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(21, 13, 1, stride=1, bias=False)
    def forward(self, x):
        l1 = self.conv_t(x)
        l2 = l1 > 0
        l3 = l1 * 0.833
        l4 = torch.where(l2, l1, l3)
        return l4
# Inputs to the model
x = torch.randn(2, 21, 21, 5)
