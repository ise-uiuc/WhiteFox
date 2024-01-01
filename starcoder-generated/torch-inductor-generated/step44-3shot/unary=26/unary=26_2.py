
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(57, 44, bias=False)
        self.conv_t = torch.nn.ConvTranspose1d(44, 19, 5, padding=0, bias=False)
    def forward(self, x4):
        m1 = self.linear(x4)
        m2 = self.conv_t(m1).clamp(min=0) * 0.089302203 + 0.37970184
        return torch.sigmoid(m2)
# Inputs to the model
x4 = torch.randn(6, 57, device='cuda')
