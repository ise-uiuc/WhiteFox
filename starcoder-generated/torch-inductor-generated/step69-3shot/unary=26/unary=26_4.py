
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 1, 128, stride=9, padding=8, groups=5, bias=False) 
        self.bias = torch.nn.Parameter(torch.Tensor(5))
    def forward(self, input):
        r1 = self.conv_t(input)
        b1 = r1 + self.bias
        e2 = torch.where(b1>1000, b1, b1*2 + 0.001)
        return math.sqrt(2.), e2
# Inputs to the model
input = torch.Tensor(1, 1, 7, 7)
