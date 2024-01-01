
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 178, 3)
        self.conv_t2 = torch.nn.ConvTranspose2d(178, 475, 4, stride=3)
    def forward(self, k):
        l1 = self.conv_t1(k)
        l2 = l1 > 0
        l3 = l1 * 0.469
        l4 = torch.where(l2, l1, l3)
        l5 = self.conv_t2(l4)
        l6 = l5 > 0
        l7 = l5 * 0.607
        l8 = torch.where(l6, l5, l7)
        return l8
# Inputs to the model
k = torch.randn(1, 2, 6, 3)
