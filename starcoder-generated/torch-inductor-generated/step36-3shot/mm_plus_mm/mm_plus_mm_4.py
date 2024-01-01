
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, t1, t2, t3, t4):
        v1 = torch.mm(t1, t2)
        v2 = torch.mm(t3, t4)
        v3 = v1 + v2
        return v3
# Inputs to the model
t1 = torch.randn(600, 1)
t2 = torch.randn(600, 1)
t3 = torch.randn(600, 1)
t4 = torch.randn(600, 1)
