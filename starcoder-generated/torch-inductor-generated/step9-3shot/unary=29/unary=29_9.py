
class Model(torch.nn.Module):
    def __init__(self, min_value= 1, max_value= 3):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(3, 8, 1, stride=1)
        self.clamp_min = torch.nn.functional.hardtanh
        self.clamp_max = torch.nn.functional.hardtanh
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, X1):
        v1 = self.conv_transpose(X1)
        v2 = self.clamp_min(v1, self.min_value, self.min_value)
        v3 = self.clamp_max(v2, self.max_value, self.max_value)
        return v3
# Inputs to the model
X1 = torch.randn(1, 3, 64)
