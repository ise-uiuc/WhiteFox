
class SinkRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # A pointwise unary operator that is supported for auto-scheduler to detect.
        # In this test case the operator is "tanh".
        x = torch.tanh(torch.cat(x, dim=1).view(x.shape[0], -1))
        return x
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
