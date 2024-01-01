
class my_view(torch.nn.Module):
    def forward(self, x):
        return x.view(1, 28, 28)
# Inputs to the model
x = torch.randn(32, 10)
