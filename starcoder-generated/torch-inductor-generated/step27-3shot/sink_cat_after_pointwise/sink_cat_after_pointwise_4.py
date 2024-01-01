
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 20, 1)
        z = x.relu()
        z = z.squeeze(dim=1).view(z.shape[0], -1)
        z = torch.tanh(z)
        z = z.detach()
        return z.contiguous()
# Inputs to the model
x = torch.randn(5, 1, 4)
