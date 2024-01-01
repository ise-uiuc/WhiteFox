
class Model(torch.nn.Module):
    def __init__(self, min_value=0.001992860070841902216662260702750233040593813455801953930408719129694538, max_value=0.0332172366312394687438425391256697464781589118424441232787068514064629444):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp(v1, self.min_value, self.max_value)
        v3 = self.tanh(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
