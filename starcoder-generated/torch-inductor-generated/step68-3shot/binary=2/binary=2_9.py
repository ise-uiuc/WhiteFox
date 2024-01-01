
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The shape for the kernel and strides should be related by a power of two
        self.conv = torch.nn.Conv2d(4, 6, 2, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        # Since it is not guaranteed that the value 256 would be exactly representable in float type
        # Use torch.Tensor.to() to make sure the value has the right type
        v2 = v1 - torch.Tensor([256]).to(v1.dtype).to(v1.device)
        return v2
# Inputs to the model
x1 = torch.randn(1, 4, 128, 64)
