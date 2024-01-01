
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = nn.Conv2d()

        v4 = x1 * v1
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 8, 8)
