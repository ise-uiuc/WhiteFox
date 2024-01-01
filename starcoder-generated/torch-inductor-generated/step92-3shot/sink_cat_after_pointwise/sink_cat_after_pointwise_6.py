
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.cat([x.flatten(1), x.flatten(1), x.flatten(1), x.flatten(1), x.flatten(1)], dim=-1).view(x.shape[0], -1)
        if (x.shape[-1]-1):
            x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
