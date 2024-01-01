
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(10, 9, 3, stride=1, padding=2, output_padding=11, bias=False)
    def forward(self, x18):
        x19 = self.conv_t(x18)
        a2 = -0.564693
        x20 = x19 * a2
        x21 = torch.flatten(x20, start_dim=0, end_dim=-1)
        y = torch.nn.functional.softmax(x21)
        return y
# Inputs to the model
x18 = torch.randn(15, 10, 55, 82)
