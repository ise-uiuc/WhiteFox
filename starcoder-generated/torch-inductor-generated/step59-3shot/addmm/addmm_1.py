
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        input1_t = torch.transpose(x1, 0, 1)
        input1_t1 = torch.add(input1_t, x2)
        v1 = torch.mm(input1_t1, x1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
