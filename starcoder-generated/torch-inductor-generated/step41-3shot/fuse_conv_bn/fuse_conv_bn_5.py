
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=1, dilation=2)
        # Note: Setting a different value for the bias here. This is for demonstrating what happens if
        # the bias is set to a non-zero value in an fused node. In this case, the bias add will be
        # replaced by an extra conv node in the fused nodes. This is to make the resulting model match
        # the eager mode graph.
        self.conv.bias = torch.nn.Parameter(torch.ones([1]))
    def forward(self, input_x):
        out = self.conv(input_x)
        return out
# Inputs to the model
x = torch.ones(1, 1, 2, 2)
