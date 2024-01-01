
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(185, 184, 12, kernel_size=(7, 2), padding=2, output_padding=0, stride=3, bias=True)
    def forward(self, x11):
        q1 = self.conv_t(x11)
        q2 = q1 > 0
        q3 = q1 * -0.5
        q4 = torch.where(q2, q1, q3)
        return (torch.nn.functional.max_pool2d(q4, (2, 1)))
# Inputs to the model
x11 = torch.randn(5, 185, 25, 34)
