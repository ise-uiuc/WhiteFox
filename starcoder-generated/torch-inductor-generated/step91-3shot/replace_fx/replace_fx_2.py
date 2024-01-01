
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=0.3)
        t1 = torch.nn.functional.dropout(x, p=0.3)
        t2 = torch.nn.functional.dropout(x, p=0.3)
        t3 = torch.nn.functional.dropout(x, p=0.3)
        t4 = torch.nn.functional.gumbel_softmax(t1, tau=1.0)
        t5 = torch.nn.functional.gumbel_softmax(t2, tau=1.0)
        t6 = torch.nn.functional.gumbel_softmax(t3, tau=1.0)
        t7 = torch.nn.functional.gumbel_softmax(t4, tau=1.0)
        t8 = torch.nn.functional.gumbel_softmax(t5, tau=1.0)
        t9 = torch.nn.functional.gumbel_softmax(t6, tau=1.0)
        return t1, t3, t7, t7, t9
# Inputs to the model
x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
