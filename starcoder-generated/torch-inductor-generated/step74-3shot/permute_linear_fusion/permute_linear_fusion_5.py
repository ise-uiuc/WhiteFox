
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        z = torch.rand(1, 3, 224, 224)
        y = torch.sigmoid(x1) # This is a typical layer invocation.
        v1 = torch.nn.functional.conv2d(y, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding)
        v2 = v1['output']
        v3 = torch.nn.functional.interpolate(v2, z.shape[-2:], mode='bilinear')
        v4 = torch.nn.functional.interpolate(v2, size=(4, 6), mode='bilinear')
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
