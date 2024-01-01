
class RemoveViewRelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = torch.relu(x)
        x = x.tanh()
        x = x.view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(3, 2, requires_grad=True)
