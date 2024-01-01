
class Model(torch.nn.Module):
    def __init__(self, min_value=-733, max_value=-394):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, bias=False,
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
