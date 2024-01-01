
input_batch = torch.zeros(1, 3, 71, 71)
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 3, 2, stride=2, padding=1, dilation=1, groups=3)
    def forward(self, input_batch):
        v1 = self.conv_transpose(input_batch)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        return v6
# Inputs to the model
x1 = torch.tensor(input_batch)
