
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose1d(1024, 400, 3, stride=2, padding=0)
        self.conv1d_1 = torch.nn.Conv1d(400, 1024, 1, stride=1, padding=0)
        self.gelu = torch.nn.GELU()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.gelu(self.conv1d_1(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 1024, 2)
