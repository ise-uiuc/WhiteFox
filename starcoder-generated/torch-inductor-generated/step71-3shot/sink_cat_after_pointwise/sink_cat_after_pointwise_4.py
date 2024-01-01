
class SinkCatAfterTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x], dim=1)
        x = x.view(int(x.shape[0]), -1)
        x = torch.tanh(x)
        x = torch.relu(x)
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
