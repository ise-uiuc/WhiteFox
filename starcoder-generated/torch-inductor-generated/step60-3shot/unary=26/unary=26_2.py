
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose1d(58, 33, 9, bias=False)
    def forward(self, x):
        y = self.conv_t(x)
        z = y > 0
        w = y * -0.963
        v = torch.where(z, y, w)
        return torch.nn.functional.relu(torch.nn.functional.max_pool2d(v, (3, 3)), 0.175208885)
# Inputs to the model
x = torch.randn(6, 40, 58, device='cuda')
