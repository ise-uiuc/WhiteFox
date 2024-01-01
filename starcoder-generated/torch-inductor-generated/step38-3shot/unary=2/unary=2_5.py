
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(2, 10, (1, 3), kernel_size=(3, 2), stride=1, padding=(1, 0), output_padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        return v1
# Inputs to the model
x1 = torch.randn(8, 2, 4)
