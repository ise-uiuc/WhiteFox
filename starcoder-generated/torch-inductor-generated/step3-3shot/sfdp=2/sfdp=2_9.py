
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, __input1__, __input2__):
        w1 = __input1__.transpose(-2, -1).matmul(__input2__)
        w2 = w1.sum(dim=2)
        w3 = w2.div(8)
        w4 = torch.nn.functional.softmax(w3, dim=1)
        w5 = torch.nn.functional.dropout(w4, p=0.3)
        w6 = w5.matmul(__input2__)
        return w6

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 8, 5)
x2 = torch.randn(1, 8, 8)
output = m(x1, x2)

