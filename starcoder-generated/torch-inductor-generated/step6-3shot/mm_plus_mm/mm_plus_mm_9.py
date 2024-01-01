
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = t1.reshape(-1)
        t3 = torch.mm(t2, t2)
        output = t3
        return output
# Inputs to the model
input = torch.randn(3, 3)
