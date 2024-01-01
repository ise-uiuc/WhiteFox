
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        x = (x1 + x2).sum()
        x *= torch.sigmoid(x)
        x *= torch.tanh(x)
        x *= (x1 + x2).sum()
        x *= torch.sigmoid(x)
        x *= torch.tanh(x)
        x *= (x1 + x2).sum()
        x *= torch.sigmoid(x)
        x *= torch.tanh(x)
        x *= (x1 + x2).sum()
        x *= torch.sigmoid(x)
        x *= torch.tanh(x)
        x *= (x1 + x2).sum()
        return x
# Inputs to the model
x1 = torch.randn(5)
x2 = torch.randn(3)
