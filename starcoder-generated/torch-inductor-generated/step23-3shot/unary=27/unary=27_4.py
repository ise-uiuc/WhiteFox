
class Model(torch.nn.Module):
    def __init__(self, min_value=0.3, max_value=0.2, p=0.01):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=0.01)
        self.conv = torch.nn.Conv2d(8, 8, 9, stride=1, padding=9)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(self.dropout(x1))
        v2 = torch.clamp(v1, min=self.min_value, max=self.max_value)
        v3 = torch.abs(v2)
        return v3
min_value = 0.3
max_value = 0.2
p = 0.01
# Inputs to the model
x1 = torch.randn(2, 8, 64, 64)
