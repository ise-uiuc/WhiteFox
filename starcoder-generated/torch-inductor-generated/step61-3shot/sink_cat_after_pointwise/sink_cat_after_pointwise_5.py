
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.randn(4, 6)
    def forward(self, x):
        y = self.weight.clone().unsqueeze(0).expand(x.shape[0], -1, -1)
        x = torch.cat((x, x), dim=1)
        x = x + y.tanh()
        print(x.shape)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
