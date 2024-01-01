
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        t1 = input.transpose(1,0)
        t2 = torch.mv(input, t1)
        output = input.mv(t2)
        return output
# Inputs to the model
input = torch.randn(2, 2)
