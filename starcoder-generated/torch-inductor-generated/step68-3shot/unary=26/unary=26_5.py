
class Model(torch.nn.Module):
    def __init__(self, input_shape=(1, 1, 24, 108), negative_slope=0):
        super().__init__()
        padding = (input_shape[-1] // 2, - input_shape[-1] // 2)
        self.upsample = torch.nn.Conv2d(
            in_channels=input_shape[0],
            out_channels=input_shape[-1],
            kernel_size=(1, 2),
            stride=(1, 2),
            padding=padding,
            bias=False,
        )
        self.pointwise_conv = torch.nn.Conv2d(
            in_channels=input_shape[2] if len(input_shape) == 4 else 1,
            out_channels=input_shape[3] if len(input_shape) == 4 else 1,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )
        self.negative_slope = negative_slope
        self.activation = torch.nn.LeakyReLU()
    def forward(self, x3):
        v0 = self.upsample(x3)
        v1 = self.pointwise_conv(v0)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v1, v3)
        return self.activation(v4)
negative_slope = 0.09
input_shape = [1, 8, 128, 768]
# Inputs to the model
x3 = torch.randn(32, *input_shape)
