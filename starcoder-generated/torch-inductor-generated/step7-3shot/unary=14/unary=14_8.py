
class Model(torch.nn.Module):
    def __init__(self):
	super().__init__()
        self.linearconv2d = torch.nn.Linear(in_features=64,
                out_features=32, bias=True)
        self.deconv = torch.nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1)
    def forward(self, x1, x2):
        v1 = self.linearconv2d(x1)
        v2 = v1.mm(x2)
        v3 = torch.sigmoid(v2)
        v4 = v2 * v3
        v5 = v1.mm(v4)
        v6 = self.deconv(v5)
        return v6
# Inputs to the model
x1 = torch.randn(64, 64)
x2 = torch.randn(64, 64)
