
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 1, stride=1000000, padding=0)
    def forward(self, inputs):
        x = self.conv(inputs)
        return x
# Inputs to the model
input_shape = (2048, 245, 245)
x = torch.randn(2, 1, *input_shape)
print(x.shape) # prints torch.Size([2, 1, 2048, 245, 245])
