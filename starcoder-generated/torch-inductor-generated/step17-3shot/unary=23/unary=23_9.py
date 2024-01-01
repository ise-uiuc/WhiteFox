
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(n_features*8)
        self.conv_transpose = torch.nn.ConvTranspose2d(8, 64, 5, stride=1, padding=0, bias=True)
        self.softmax = torch.nn.Softmax(dim=-1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v1 = v1.view(int(v1.size(0)*v1.size(1)*v1.size(2)), v1.size(-1))
        v2 = self.softmax(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 5, 5)
