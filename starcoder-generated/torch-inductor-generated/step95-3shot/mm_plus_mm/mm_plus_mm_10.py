
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4):
        m1 = torch.mm(x1, x2) # Matrix multiplication
        m2 = torch.mm(x3, x4) # Matrix multiplication
        m3 = m1 + m2 # Addition
        return m3
# Inputs to the model
x1 = torch.randn(64,64)
x2 = torch.randn(64,64)
x3 = torch.randn(64,64)
x4 = torch.randn(64,64)
