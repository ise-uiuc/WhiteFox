
class Model(torch.nn.Module):
    def __init__(self, min_value='string', max_value='string_2'):
        print(f'Input value of min_value: {min_value}, Input value of max_value: {max_value}. String values set by default.')
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = v1.clamp(self.min_value, self.max_value)  # this line can throw error - need to fix
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
min_value = 4.0
max_value = 1.4
x1 = torch.randn(1, 3, 64, 64)
