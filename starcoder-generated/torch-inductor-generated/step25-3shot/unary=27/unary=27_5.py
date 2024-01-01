
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 6, 6, stride=1, padding=1)
        self.t1 = torch.nn.ReLU()
        self.min = min
        self.max = max
    def forward(self, input, min, max):
        v0 = torch.zeros_like(input.mul(0))
        t1 = v0
        conv_input = t1.mul(0) + input
        v11 = self.conv(conv_input)
        relu_input = torch.clamp_max(v11, 0)
        v12 = self.t1(relu_input)
        v2 = torch.clamp_max(v12, max)
        v3 = v2.mul(0) + input
        return v3
min = 2
max = 1
# Inputs to the model
input = torch.randn(1, 3, 52, 52)
min = 2
max = 1
