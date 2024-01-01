
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t = torch.rand_like(x)
        x = F.dropout(x, p=0.5)
        return x + t
# Inputs to the model
x = torch.rand(1, 1, 2)
