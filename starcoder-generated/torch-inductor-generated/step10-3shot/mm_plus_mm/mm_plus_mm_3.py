
class Model(torch.nn.Module):
    def forward(self, x):
        a = torch.matmul(x, x)
        return a
# Inputs to the model
x = torch.randn(4, 4)
