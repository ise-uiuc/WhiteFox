
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t1, t2):
        t3 = torch.mm(t2, t1)
        return t3
# Inputs to the model
t1 = torch.randn(3, 3, requires_grad=True)
t2 = torch.randn(3, 3, requires_grad=True)
