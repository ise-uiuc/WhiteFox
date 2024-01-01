
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 6, 5, stride=1, padding=0, bias=False)
        self.act = torch.nn.PReLU(6)
    def forward(self, x6):
        z5 = self.conv_t(x6)
        z6 = z5 > 0
        z7 = z5 * 0.995
        z8 = torch.where(z6, z5, z7)
        z9 = self.act(z8)
        return torch.nn.functional.log_softmax(torch.nn.functional.layer_norm(z9, [1, 1, 24, 35]))
# Inputs to the model
x6 = torch.randn(3, 1, 65, 24)
