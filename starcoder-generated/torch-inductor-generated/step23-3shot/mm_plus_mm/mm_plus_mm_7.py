
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        v2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        v4 = v1 + v2 + v3
        return t1 + t2 + t3 + v4
# Inputs to the model
input = torch.randn(1, 10)
