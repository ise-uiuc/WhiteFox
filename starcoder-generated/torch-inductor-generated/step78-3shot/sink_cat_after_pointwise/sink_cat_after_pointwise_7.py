
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=1)
        x = x[:, x.shape[-1] // 2:, :]
        return torch.cat((x, x), dim=-1)
# Inputs to the model
x = torch.randn(2, 3, 4, 5)
