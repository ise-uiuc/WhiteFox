
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = t1 + t2 * t3
        return t4
# Inputs to the model
input = torch.randn(8, 8)
