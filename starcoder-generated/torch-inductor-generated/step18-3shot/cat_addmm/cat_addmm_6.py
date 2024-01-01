
class Model(nn.Module):
    def forward(self, x):
        x = torch.randn(2, 4)
        x = torch.randn(2, 6)
        y = torch.cat((x, x), dim=1)
        return y
# Inputs to the model
x = torch.randn(3)
