
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(12, 128, 5, padding=2, output_padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.view(8, 128, -1)
        v3 = torch.norm(v2, p=1, dim=2, keepdim=True)
        v4 = torch.tanh(v3)
        v5 = torch.relu(v4)
        v6 = v5.contiguous().view(1, -1)
        return v6
# Inputs to the model
x1 = torch.randn(1, 30, 16)
