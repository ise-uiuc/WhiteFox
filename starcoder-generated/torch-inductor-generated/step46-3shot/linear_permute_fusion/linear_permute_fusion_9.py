
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1
        v3 = x1.shape
        v2 = torch.nn.functional.linear(v1, v3, v3, True)
        v3 = v2.permute(0, 2, 1)
        return v3
# Inputs to the model
x1 = torch.randn(4, 3, 4)
