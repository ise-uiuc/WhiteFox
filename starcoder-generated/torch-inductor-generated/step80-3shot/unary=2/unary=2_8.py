 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 6, (2, 1), stride=(2, 1))
        self.conv_transpose2 = torch.nn.ConvTranspose2d(6, 3, (1, 1), stride=(1, 1))
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = self.conv_transpose2(v1)
        n2 = 1*6+2
        v3 = v2 * 0.5
        v4 = v2 * v2 * v2
        v5 = v4 * 0.044715
        v6 = v2 + v5
        v7 = v6 * 0.7978845608028654
        v8 = torch.tanh(v7)
        v9 = v8 + 1
        v10 = v3 * v9
        n4 = 1*6+1
        return v10
# Inputs to the model
x1 = torch.randn(1, 3, 2, 1)
