
class Model(torch.nn.Module):
    def forward(self, x, y):
        return torch.mm(x, x) + torch.mm(x, y) + torch.mm(y, x) + torch.mm(y, y)
# Inputs to the model
x = torch.randn(128, 17)
y = torch.randn(17, 128)
