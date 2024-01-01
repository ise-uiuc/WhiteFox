
class Model(torch.nn.Module):
    def __init__(self, conv_num, kernel_size, stride, padding, input_channels, dilation, kernel, bias=True):
        super().__init__()
        self.layers = [
        torch.nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, dilation, 1, bias)
        for _, output_channels in zip(range(conv_num), kernel)
    ]
        self.layers = torch.nn.ModuleList(self.layers)
    def forward(self, x1):
        [y for y in [x(x1) for x in self.layers if'relu' in x.__class__.__name__.lower()]]
        return x1 * 0.99 + 0.99
# Inputs to the model
x1 = torch.randn(5, 2, 1, 1)
