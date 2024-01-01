
# ReLU6 with input and output shapes can be different
# Input size of the model is 224 * 224 * 3
# Output size of the model is 223 * 223 * 8
# Filter size is 3
# Stride is 2
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp(v1, 0, 255) # ReLU6
        v3 = v2.permute(0, 2, 3, 1) # Permute the tensor to NCHW
        return v3
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
