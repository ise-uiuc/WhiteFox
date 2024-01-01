
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convt2d_0 = torch.nn.ConvTranspose2d(6, 1, (1, 16), stride=(1, 1))
        self.tanh2_0 = torch.nn.Tanh()
    def forward(self, x0):
        x1 = self.convt2d_0(x0)
        x2 = torch.nn.functional.gelu(x1)
        x3 = self.tanh2_0(x2)
        x4 = x0 + x3
        return x4
# Inputs to the model
x0 = torch.randn(1, 6, 128, 32)
