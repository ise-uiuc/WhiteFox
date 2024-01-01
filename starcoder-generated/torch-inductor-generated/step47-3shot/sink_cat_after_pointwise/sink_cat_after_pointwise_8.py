
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, torch.zeros(2, 3, 4)), dim=1)
        y = y.view(y.shape[0], -1) if y.shape[1]!= 4 else y.view(y.shape[0], -2)
        y = y.view(y.shape[0], -1)
        return torch.where((x < 0.0), tensor1, tensor2)
# Inputs to the model
x = torch.randn(2, 3, 4)
