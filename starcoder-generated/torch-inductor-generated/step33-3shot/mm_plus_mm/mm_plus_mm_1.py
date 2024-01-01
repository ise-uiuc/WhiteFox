
class Model(torch.nn.Module):
    def forward(self, x):
        x2 = torch.mm(x, x)
        x1 = torch.mm(x, x)
        x3 = x1 + x2
        return x3
# Inputs to the model
x =torch-n(6, 6)
