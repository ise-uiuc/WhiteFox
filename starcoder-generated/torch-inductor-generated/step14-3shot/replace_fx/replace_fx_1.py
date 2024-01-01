
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        x1 = torch.rand(1, 2, 2)
        x2 = torch.rand(1, 2, 2)
        x3 = torch.rand_like(x1)
        x4 = torch.rand(1, 2, 2)
        x5 = torch.zeros(1, 2, 2)
        x6 = torch.rand_like(x1)
        return x6
# Inputs to the model
