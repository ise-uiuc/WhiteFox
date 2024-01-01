
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(305, 187, 2, stride=3, padding=1, bias=False)
    def forward(self, x):
        g1 = self.conv_t(x)
        g2 = g1 > 0
        g3 = g1 * -1.5
        g4 = torch.where(g2, g1, g3)
        return torch.nn.functional.hardsigmoid(torch.nn.functional.elu(torch.nn.functional.relu(g4))) # hardsigmoid followed by elu
# Inputs to the model
x = torch.randn(35, 305, 98, 9)
