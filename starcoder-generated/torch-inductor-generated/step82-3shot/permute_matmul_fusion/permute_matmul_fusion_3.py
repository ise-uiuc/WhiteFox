
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        n1 = x1.squeeze().transpose(0, 1)
        r1 = n1.squeeze()
        v1 = n1.flip(dims=(0, 1))
        t1 = n1.unsqueeze(-1)
        return x1.add_(t1)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 3)
