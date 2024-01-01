
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input.transpose(2, 1), input)
        t2 = torch.mm(input, input.transpose(2, 1))
        t3 = t1 + t2
        return t3
# Inputs to the model
input = torch.randn(100, 25, 3)
