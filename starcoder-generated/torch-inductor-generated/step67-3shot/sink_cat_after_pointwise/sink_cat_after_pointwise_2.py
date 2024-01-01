
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.relu(torch.cat([x, torch.zeros(x.shape)], dim=-1))
        return x
# Inputs to the model
x = torch.randn(1, requires_grad=True)
