
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=0.3)
        t2 = torch.nn.functional.gumbel_softmax(t1, tau=1.0)
        return t2
# Inputs to the model
x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
