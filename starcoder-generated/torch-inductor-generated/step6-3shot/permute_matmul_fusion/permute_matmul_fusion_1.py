
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.add(x1.permute(1, 0), -x2.permute(1, 0), alpha=1) + 1
# Inputs to the model
x1 = torch.randn(1, 20)
x2 = torch.randn(1, 20)
