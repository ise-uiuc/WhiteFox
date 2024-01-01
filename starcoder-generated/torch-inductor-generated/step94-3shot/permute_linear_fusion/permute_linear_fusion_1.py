
class Model(torch.nn.Module):
    def forward(self, x1):
        v1 = x1.permute(1, 0)
        return v1
# Inputs to the model
x1 = torch.randn(2, 1)
