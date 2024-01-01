
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((torch.cat((x, x), dim=1), x), dim=0)
        x = x.transpose(1, 0).view(-1, x.shape[2:] * 4)
        return x
# Inputs to the model
x = torch.randn(3, 4, 3)
