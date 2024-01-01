
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 1024)
 
    def forward(self, nchw_input):
        nhwc_input = nchw_input.permute((0, 2, 3, 1))
        result = self.linear(nhwc_input)
        result = result.permute((0, 3, 1, 2))
        negative_slope = 0.1
        lrelu = torch.nn.LeakyReLU(negative_slope)
        return lrelu(result)

# Initializing the model
m = Model()

# Input to the model
nchw_input = torch.randn(1, 512, 8, 7)
