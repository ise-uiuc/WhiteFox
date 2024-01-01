
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        t1 = torch.rand((x1.shape[0], 2))
        t2 = torch.rand((x1.shape[0], 5), requires_grad=False)
        for e1, e2 in zip(t1,t2):
            x1 = torch.nn.functional.dropout(x1, p=0.5)
            x1 = torch.mean(x1)
        return x1
# Inputs to the model
x1 = torch.randn(12)
