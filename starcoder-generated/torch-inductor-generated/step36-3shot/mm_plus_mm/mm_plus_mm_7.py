
class Model(torch.nn.Module):
    def forward(self, x):
        x_ = None
        torch.mm(x, x, out=x_)
        return x
# Inputs to the model
x = torch.randn(5, 5)
