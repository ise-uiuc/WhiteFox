
class Model(torch.nn.Module):
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mm(x, x)
        return t1 + t2
# Inputs to the model
x = torch.randn(20, 20)
