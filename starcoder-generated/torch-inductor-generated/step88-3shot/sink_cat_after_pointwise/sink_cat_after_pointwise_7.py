
class SinkTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.tanh() if x.shape!= (1, 4) else x.tanh()
        return x
# Inputs to the model
x = torch.randn(1, 4, requires_grad=True)
