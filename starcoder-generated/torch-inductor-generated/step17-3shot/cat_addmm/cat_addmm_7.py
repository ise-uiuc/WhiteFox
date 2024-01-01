
class Model(nn.Module):
    def __init__(self, input_channels, output_channels, kernel):
        super().__init__()
        self.layers = nn.Conv2d(input_channels, output_channels, kernel)
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
input_channels = 3
output_channels = 6
kernel = 2
x = torch.randn(20, 3, 299, 299)
