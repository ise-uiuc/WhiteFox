
class Model(torch.nn.Module):
    def __init__(self, min_value=0.4, max_value=1):
        super().__init__()
        self.conv_transpose_1 = torch.nn.ConvTranspose1d(1, 16, 1, stride=1, padding=1)
        self.conv_transpose_2 = torch.nn.ConvTranspose1d(16, 1, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose_1(x1)
        v2 = self.conv_transpose_2(v1)
        v3 = torch.flatten(v2) 
        v2_a = torch.clamp(v1, self.min_value, self.max_value)
        v2_b = torch.clamp(v2, self.min_value, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 128)
