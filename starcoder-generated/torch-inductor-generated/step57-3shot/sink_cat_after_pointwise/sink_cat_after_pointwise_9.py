
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x, x), dim=0)
        x = x.view(-1, x.shape.numel() // x.shape[0])
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
