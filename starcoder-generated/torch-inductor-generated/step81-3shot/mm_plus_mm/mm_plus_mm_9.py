
class Model(nn.Module):
    def forward(self, x1, x2):
        z1 = x1 + x2
        z2 = x2 + x1
        out1 = torch.tanh(x2)
        out2 = x1 + x2
        return (z1, z2, out1, out2)
# Inputs to the model
x1 = torch.randn(3, 4)
x2 = torch.randn(3, 4)
