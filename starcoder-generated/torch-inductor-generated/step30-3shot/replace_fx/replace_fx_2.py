
class Test(torch.nn.Module):
    def forward(self, x):
        x = torch.rand(1)
        return x
# Inputs to the model
x = torch.randn(3)
