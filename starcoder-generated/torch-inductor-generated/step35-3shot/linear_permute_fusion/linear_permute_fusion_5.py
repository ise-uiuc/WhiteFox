
class Model(torch.nn.Module):
    def forward(self, x1):
        v3 = torch.nn.functional.dropout(x1)
        v1 = v3.permute(0, 1, 3, 2)
        return v1
# Inputs to the model
x1 = torch.randn(1, 2, 2, 2)
