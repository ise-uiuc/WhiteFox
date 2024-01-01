
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(2044, 32, 3, stride=32)
        self.batchnorm2d = torch.nn.BatchNorm2d(num_features=24, eps=0.0010000000475, momentum=0.02, affine=False, track_running_stats=False)
    def forward(self, input, input1):
        v1 = self.conv2d(input)
        v2 = self.batchnorm2d(v1)
        v2.contiguous()
        v3 = v2 * 0.5
        v4 = v2 * v2
        v5 = v4 * v2
        v6 = v5 * 0.044715
        v7 = v2 + v6
        v8 = v7 * 0.7978845608028654
        v9 = torch.tanh(v8)
        v10 = v9 + 1
        v11 = v3 * v10
        return v11
# Inputs to the model
input = torch.randn(1, 2044, 76, 76)
input1 = torch.randn(5)
