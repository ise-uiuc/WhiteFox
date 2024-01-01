
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(16, 68, 2, stride=1, padding=2, output_padding=1, dilation=0)
    def forward(self, x6):
        l1 = self.conv_t(x6)
        l2 = l1 > 0
        l3 = l1 * -0.3757
        l4 = torch.where(l2, l1, l3)
        return torch.nn.functional.relu(l4)
# Inputs to the model
x6 = torch.randn(30, 16, 29, 21)
