
class Model(nn.Module):
    def forward(self, x1, x2):
        x3 = x1 + x2
        x4 = x1 * x2
        return x3, x4
# Inputs to the model
x1 = torch.randn(32, 3, 224, 224)
x2 = torch.randn(3, 32, 3, 224, 224)
