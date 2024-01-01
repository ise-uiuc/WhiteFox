
class model_torchtrace(nn.Module):
    def forward(self, x1, x2, x3):
        x = (x1 * x2) * x3
        return x
# Inputs to the model
x1 = torch.randn(1, 300, 300)
x2 = torch.randn(1, 300, 300)
x3 = torch.randn(1, 300, 300)
