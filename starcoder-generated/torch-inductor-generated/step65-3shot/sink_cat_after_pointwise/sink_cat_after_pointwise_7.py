
class SinkCatAfterRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.cat([x, x], dim=1)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
