
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, (1, 3), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.tanh(v1)
        v3 = torch.nn.functional.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
