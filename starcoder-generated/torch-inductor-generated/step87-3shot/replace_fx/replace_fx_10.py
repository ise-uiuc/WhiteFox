
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x3 = torch.nn.functional.dropout(x1, p=0.5)
        x4 = torch.nn.functional.dropout(x2, p=0.2)
        x5 = (x3 * x4).sum(-1)
        return x5
# Inputs to the model
x1 = torch.zeros([1,3,3], requires_grad=True)
x2 = torch.zeros([1,3,3], requires_grad=True)
