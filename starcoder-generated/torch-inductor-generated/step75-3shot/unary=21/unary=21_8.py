
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AvgPool1d(3, stride=1, padding=1, count_include_pad=True)
    def forward(self, x13):
        v4 = self.pool(x13)
        v5 = torch.tanh(v4)
        return v5
# Inputs to the model
x13 = torch.randn(1, 1, 5)
