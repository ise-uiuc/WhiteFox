
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x1 = F.dropout(x1, p=0.3)
        t1 = torch.nn.functional.dropout(x1, p=0.3)
        return t1
# Inputs to the model
x1 = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
