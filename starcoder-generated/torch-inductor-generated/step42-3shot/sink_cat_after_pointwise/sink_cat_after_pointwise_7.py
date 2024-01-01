
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x.reshape(-1, 8)
        return x.squeeze().view(-1)
# Inputs to the model
x = torch.randn(2, 2, 2)
