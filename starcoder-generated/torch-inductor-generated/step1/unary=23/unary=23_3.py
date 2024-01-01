
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, output_padding=1, padding=1)
 
    def forward(self, x):
        v1 = self.deconv(x)
        v2 = torch.tanh(v1)
        return v2