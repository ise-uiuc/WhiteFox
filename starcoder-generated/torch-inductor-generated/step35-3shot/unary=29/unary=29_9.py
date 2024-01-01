
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.610921458183803, max_value=-2.613650681131714):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose2d(9, 12, 2, stride=2)
        self.conv_transpose_2 = torch.nn.ConvTranspose2d(12, 33, 1, stride=1)
        self.conv_transpose_3 = torch.nn.ConvTranspose1d(33, 8, 1, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = self.conv_transpose_3(v2)
        v3_a = torch.clamp(v2, self.min_value, self.max_value)
        v3_b = torch.clamp(v3, self.min_value, self.max_value)
        return v3_b
# Inputs to the model
x1 = torch.randn(1, 9, 11, 11)
