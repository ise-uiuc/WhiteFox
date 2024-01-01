
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(128, 128, 5, stride=3, padding=0, bias=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v_relu = self.relu(v1)
        v2 = v_relu + 3
        v3 = torch.clamp(v2, min=0)
        v4 = torch.clamp(v3, max=6)
        v5 = v_relu * v4
        v6 = v5/ 6
        return v6
# Inputs to the model
x1 = torch.randn(1, 128, 32, 32)
