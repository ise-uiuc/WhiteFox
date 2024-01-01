
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(110, 91, 13, bias=False, padding=0, stride=9)
    def forward(self, input):
        x34 = self.conv_t(input)
        x35 = x34 > 0
        x36 = x34 * -1.0368
        x37 = torch.where(x35, x34, x36)
        return torch.flatten(x37, start_dim=2, end_dim=3)
# Inputs to the model
input = torch.randn(1, 110, 15, 166)
