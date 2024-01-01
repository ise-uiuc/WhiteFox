
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        return t1 + t2
# Input to the model
input = torch.randn(4, 4)
