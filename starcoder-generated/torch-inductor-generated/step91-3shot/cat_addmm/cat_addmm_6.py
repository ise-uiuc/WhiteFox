
class Model(nn.Module):
    def forward(self, x):
        x = torch.squeeze(x, dim=1)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
