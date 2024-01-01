
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(128, 128, kernel_size=(3, 1),stride=1,padding=0,bias=False)
    def forward(self, input):
        v1 = self.conv_t(input)
        v2 = v1 > 0
        v3 = v1 * -0.2171
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
input = torch.randn(3, 128, 4, 5)
