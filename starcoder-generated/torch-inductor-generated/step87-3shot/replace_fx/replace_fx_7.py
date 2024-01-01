
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = torch.nn.functional.dropout(x1, p=0.5)
        x2 = torch.rand_like(x1)
        x3 = torch.rand_like(x1)
        a2 = x2 * x1
        return a2.sum(-1)
# Inputs to the model
x1 = torch.zeros([1, 3, 3])   
