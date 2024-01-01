
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x1)
        m_res = v1.mean()
        return m_res
# Inputs to the model
x1 = torch.randn(1321, 1321)
x2 = torch.randn(1, 1)
