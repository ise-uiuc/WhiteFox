
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x], dim=1).view(-1, 200)
        x = x.transpose(0, 1).view(x.shape[1:], x.shape[0])
# Inputs to the model
x = torch.randn(3, 512, 16, 16, requires_grad=True)
