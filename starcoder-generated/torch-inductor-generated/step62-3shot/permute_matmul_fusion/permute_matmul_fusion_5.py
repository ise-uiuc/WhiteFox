
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        u = x1.permute(0, 2, 1)[0][0][0]
        v = x2.permute(0, 2, 1)[0][0][0]
        return u + v
# Inputs to the model
x1 = torch.tensor([[[-1., 0.]]])
x2 = torch.tensor([[[1., 2.]]])
