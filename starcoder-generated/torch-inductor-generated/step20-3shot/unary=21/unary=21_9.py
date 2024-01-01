
class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.l1 = torch.nn.Linear(400, 83)
    def forward(self, x):
        v1 = torch.tanh(self.l1(x))
        v2 = v1.t()
        return v2
# Inputs to the model
input = torch.randn(10, 400)
