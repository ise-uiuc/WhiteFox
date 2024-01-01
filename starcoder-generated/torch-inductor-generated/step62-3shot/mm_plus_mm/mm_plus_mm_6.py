
class Model(torch.nn.Module):
    def forward(self, input1):
        c1 = torch.cat([input1, input1, input1], dim=1)
        return torch.sum(c1, dim=0)
# Inputs to the model
input1 = torch.randn(16, 16)
