
class Model(torch.nn.Module):
    def forward(self, x1):
        x1 = torch.mm(x1, x1)
        x1 = torch.mm(x1, x1)
        x1 = torch.mm(x1, x1)
        x1 = torch.mm(x1, x1)
        x1 = torch.mm(x1, x1)
        x1 = torch.mm(x1, x1)
        return x1
# Inputs to the model
x1 = torch.randn(64, 64)
