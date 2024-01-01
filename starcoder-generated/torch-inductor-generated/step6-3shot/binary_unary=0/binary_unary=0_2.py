
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.relu(x1)
        v2 = x2 + v1
        return v2
# Inputs to the model
x1 = 1
x2 = torch.randn(1, 16, 64, 64)
