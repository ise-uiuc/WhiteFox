
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(527, 6, 8, stride=19, padding=0, bias=False)
    def forward(self, x2):
        g2 = self.conv_t(x2)
        g3 = g2 > 0
        g4 = g2 * 2.63601
        g5 = torch.where(g3, torch.transpose(g4, 4, 3), -1.31601 * torch.transpose(g4, 4, 3))
        return torch.transpose(g5, 4, 3)
# Inputs to the model
x2 = torch.randn(195, 527, 117, 1)
