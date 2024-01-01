
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input + 1, input)
        t2 = torch.mm(input * 2, input)
        return t1 + t2
# Inputs to the model
input = torch.randn(1, 1)
