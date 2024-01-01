
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.nn.functional.dropout(x, p=0.3)
        return x + x + x
# Inputs to the model
x = torch.Tensor([[0.25, 0.25, 0.25, 0.25]])
