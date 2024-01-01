
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=2)
        return y.view(y.shape[0], y.shape[1] * y.shape[2]).add(1.2).permute(1, 0).view(x.shape[0], -1).permute(1, 0).tanh() if y.shape[0] == 1 else y.tanh().permute(1, 0).add(1.2).view(-1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
