
class Model(torch.nn.Module):
    def forward(self, input):
        t2 = torch.mm(input, input)
        t1 = torch.mm(input, t2)
        return t1 * t2
# Inputs to the model
input = torch.randn(7, 7)
