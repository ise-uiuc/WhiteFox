
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        if x.shape[-1] >= 10:
            return torch.cat((torch.nn.functional.relu(x), x), dim=1).view((x.shape[0], -1)).relu()
        else:
            return torch.cat((x, x), dim=1).view((x.shape[0], -1))
# Inputs to the model
x = torch.randn(2, 3, 4)
