
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        x1, x2 = self.conv1(x1), self.conv2(x2)
        v1, v2 = x1, x2
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
