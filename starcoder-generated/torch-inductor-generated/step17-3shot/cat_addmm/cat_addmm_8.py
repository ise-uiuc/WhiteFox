
class Model(nn.Module):
    def forward(self, x):
        x = torch.mm(x, x)
        y = torch.cat([x], dim=1)
        z = torch.cat([y], dim=-1)
        return z
# Inputs to the model
x = torch.randn(2, 4)
