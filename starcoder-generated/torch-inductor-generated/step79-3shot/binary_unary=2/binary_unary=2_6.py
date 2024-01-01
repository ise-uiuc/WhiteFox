
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 10.0000
        v2_shape = v2.shape
        v2_shape1 = v2_shape[0]
        v2_shape2 = v2_shape[1]
        v2_shape3 = v2_shape[2] #?
        v2_shape4 = v2_shape[3] #?
        v3 = torch.squeeze(v2, 0) #?
        v4 = F.relu(v3) #?
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
