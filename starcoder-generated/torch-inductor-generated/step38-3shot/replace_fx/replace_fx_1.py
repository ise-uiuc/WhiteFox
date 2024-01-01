
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        a = torch.rand(100, requires_grad=True)  # this node will be inserted before dropout
        t1 = torch.rand_like(x1)
        x2 = F.dropout(t1)
        return x2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
