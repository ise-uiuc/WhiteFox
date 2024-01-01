
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.clamp(x, 0, 1)
        x = x + 1
        x = x * 2
        x = torch.sign(x)
        x = torch.add(x, x)
        x = torch.floor(x)
        x = torch.cat((x, x), dim = 1)
        x = torch.mul(x, x)  # TODO: Should not trigger because it is not a multiplication. Should be an element-wise multiplication.
        x = torch.sub(x, x)
        x = torch.clamp(x, -1, 0)
        x = x + 1
        x = x * 2
        x = torch.tanh(x) + x ** 2
        x = x + 1
        x = x * 2
        x = torch.tanh(x)
        x = x + 1
        x = torch.sum(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.squeeze(x, dim=1)
        x = torch.unbind(x)
        x = torch.squeeze(x, dim = 0)
        x = torch.unsqueeze(x, 0)
        x = x + 1
        x = x * 2
        return x
# Inputs to the model
x = torch.randn(2, 1, 5, 6, 7)
