
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 4, stride=1)
    def forward(self, input_1):
        v1 = self.conv(input_1)
        v2 = v1 - 260.39
        v2 = v2 - 1
        v3 = v2 - 0.5
        v4 = v3 - 228
        v5 = v4 - -0.54
        v6 = v5 - 0.9
        v7 = v6 - -0.377
        v8 = v7 - -0.78
        v9 = v8 - -1.23
        v10 = v9 - 0.3
        return v7
# Inputs to the model
input_1 = torch.randn(1, 3, 32, 32)
