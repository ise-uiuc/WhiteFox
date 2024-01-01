
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.1, max_value=-3.5):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v2 = self.conv_transpose2d(x3)
        v3 = v2 + 0.7152036530499315
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
    
# Inputs to the model
x3 = torch.randn(1, 3, 128, 114)
