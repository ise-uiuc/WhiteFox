
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        v1 = torch.mm(x2, x1)
        v2 = v1.unsqueeze(-1)
        v3 = v2.squeeze(1)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 4)
