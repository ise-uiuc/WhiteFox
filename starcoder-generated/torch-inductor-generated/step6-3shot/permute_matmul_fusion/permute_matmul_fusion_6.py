
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = torch.bmm(v1, x2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
