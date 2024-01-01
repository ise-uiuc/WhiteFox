
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(2, 2)
    def forward(self, x1):
        x2, state_h = self.lstm(x1)
        x3 = torch.rand_like(x1, requires_grad=True)
        _ = torch.cat([x2, x3], dim=1)
        return state_h._values.clone()
# Inputs to the model
x1 = torch.randn(3, 3, 2, requires_grad=True)
h0 = torch.randn(2, 1, 2, requires_grad=True)
c0 = torch.randn(2, 1, 2, requires_grad=True)
