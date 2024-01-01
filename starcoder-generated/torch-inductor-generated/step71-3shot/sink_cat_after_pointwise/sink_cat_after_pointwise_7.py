
class SinkCatAfterRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], 1, -1, 1)
        x = torch.cat([x + 1, x + 1, x + 1], dim=3)
        x = torch.relu(x)
        return x[0, :, :, 0]
# Inputs to the model
x = torch.randn(1, 3, 2, 2, requires_grad=True)
