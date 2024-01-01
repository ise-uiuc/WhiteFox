
class Model(torch.nn.Module):
    def forward(self, x, y):
        t1 = torch.cat([x, y], dim=2)
        return torch.mean(t1, dim=1)
# Inputs to the model
x = torch.randn(4,32,64)
y = torch.randn(4,32,64)
