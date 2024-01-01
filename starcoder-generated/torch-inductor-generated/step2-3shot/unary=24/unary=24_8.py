
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = x * 0.01 + x * 0.01
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
