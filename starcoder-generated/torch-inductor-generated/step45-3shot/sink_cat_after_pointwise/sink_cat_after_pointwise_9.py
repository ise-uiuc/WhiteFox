
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x[:x.shape[0]-1], x[-1].unsqueeze(dim=0)], dim=0)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
