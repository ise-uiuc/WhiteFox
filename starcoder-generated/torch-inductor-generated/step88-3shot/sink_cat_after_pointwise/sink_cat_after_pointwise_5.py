
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([torch.relu(torch.tanh(x)), torch.tanh(x)], dim=1)
        z = torch.relu(y)
        x = z if (x.shape == (3, ) and y.shape == (2, 6) and z.shape == (3, 6)) else x[:2]
        return torch.stack([x])
# Inputs to the model. Note: This input will lead the model to trigger `sink_cat_after_pointwise` pattern
x = torch.randn(6, requires_grad=True)
