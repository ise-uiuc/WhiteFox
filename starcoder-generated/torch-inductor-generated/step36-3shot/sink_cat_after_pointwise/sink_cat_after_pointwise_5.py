
class SinkRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.tanh(x)
        x = torch.relu(x) if x.shape == (1, 2) else x.relu()
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
