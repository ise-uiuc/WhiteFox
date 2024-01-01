
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(9, 9, 2, stride=1, padding=0, bias=True)
        self.layer_norm = torch.nn.LayerNorm([14,15], elementwise_affine=False)
        self.relu = torch.nn.ReLU()
    def forward(self, x2):
        x3 = self.conv_t(x2)
        u2 = self.layer_norm(x3)
        x5 = u2 <= 0.35355338454437256
        x6 = x5.int()
        x7 = x6.float()
        x8 = self.relu(x3)
        x9 = self.relu(x7)
        return u2
# Inputs to the model
x2 = torch.randn(1, 9, 14, 15)
