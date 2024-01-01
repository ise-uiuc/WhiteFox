
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y = x1.permute(0, 2, 1)
        x2 = y * 1.15
        x2 = x2.flatten(start_dim=0, end_dim=1)
        return x2
# Inputs to the model
x1 = torch.ones(1, 2, 2)
