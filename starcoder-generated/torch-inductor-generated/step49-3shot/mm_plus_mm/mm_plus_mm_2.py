
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.mm(torch.mm(x.transpose(-1, -2), x), x.transpose(-1, -2))
# Inputs to the model
x = torch.randn(3, 3)
# Inputs to the model ended
