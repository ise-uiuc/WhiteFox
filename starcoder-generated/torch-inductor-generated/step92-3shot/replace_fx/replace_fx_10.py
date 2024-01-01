
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.rand_like(x)
# Inputs to the model
x = torch.randn(2)
