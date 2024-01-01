
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(472, 470, 5, stride=3, padding=3, groups=2, bias=False)
    def forward(self, x0):
        l1 = self.conv_t(x0)
        l2 = l1 > 0
        l3 = l1 * -0.5
        l4 = torch.where(l2, l1, l3)
        return torch.nn.functional.relu(l4)
# Inputs to the model
x0 = torch.randn(56, 472, 58, 72)
