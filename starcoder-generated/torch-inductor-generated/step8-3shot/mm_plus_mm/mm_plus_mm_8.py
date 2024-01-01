
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(t1, t1)
        return t2
# Inputs to the model
input1 = torch.randn(3, 3)
