
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t1 = torch.nn.ConvTranspose2d(2, 17, 3, stride=1, padding=2, output_padding=0, groups=2, bias=True)
        self.conv_t2 = torch.nn.ConvTranspose2d(1, 9, 1, stride=1, padding=0, output_padding=0, groups=1, bias=True)
    def forward(self, x):
        x2 = self.conv_t1(x)
        x3 = self.conv_t2(x2)
        # The model will contain both conv transpose ops
        # Please use the output tensors as the input for different pointwise op
        return x3 + x2 + torch.tanh(x)
# Inputs to the model
x = torch.randn(1, 2, 67, 65)
