
class Model(torch.nn.Module):
    def __init__(self, output_channel, kernel_size, padding=0, bias=False, other=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, output_channel, kernel_size, padding=padding, bias=bias)
        self.linear = torch.nn.Linear(output_channel, 8)
        if other is None: # Use the bias of the linear transformation as an additional tensor
            self.other = self.linear.bias.detach().clone().detach()
        else:
            self.other = other
        
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 + self.other
        v3 = self.linear(v2)
        return v3

# Initializing the model
other = torch.randn(1, 8)
m = Model(3, 1, other=other)

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
