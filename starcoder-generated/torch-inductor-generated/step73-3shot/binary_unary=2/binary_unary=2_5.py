
class Model(torch.nn.Module):
    def forward(self, x):
        v1 = F.pad(x, (0, 0, 0, 0, 1, 1, 1, 1), value=1.0 - x)
        v2 = v1 - x
        v3 = F.relu(v2)
        v4 = F.pad(v3, (1, 1, 1, 1, 1, 1, 1, 1), value=2.0 - v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
