
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.test = 0
    def forward(self, x):
        x = x.view(10, 10).permute(1, 0) if x.shape!= (10, 10) else x
        out = torch.cat([x, x, x], dim=-1)
        return out
# Inputs to the model
x = torch.rand((1, 1))
y = torch.rand((1, 1))
z = torch.rand((1, 1))
