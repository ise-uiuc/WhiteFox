
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = t3 + t2 + t1
        return t3
# Inputs to the model
input = torch.randn(10, 10)
