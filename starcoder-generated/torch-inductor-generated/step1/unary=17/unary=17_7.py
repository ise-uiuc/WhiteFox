
class ReLUConvTranspose(torch.nn.Module):
    def __init__(self, C_in, C_out):
        super(ReLUConvTranspose, self).__init__()
        self.relu_conv_tranpose = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(C_in, C_out, kernel_size=3, stride=2,
            padding=1)
        )
 
    def forward(self, x):
        return self.relu_conv_transpose(x)

# Initializing the model
m = ReLUConvTranspose(3, 8)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
