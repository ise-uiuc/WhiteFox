
class Model(torch.nn.Module):
    def forward(self, input):
        t02 = torch.mm(input, input)
        t03 = torch.mm(input, input)
        t04 = torch.mm(input, input)
        t05 = torch.mm(input, input)
        t06 = torch.mm(input, input)
        return t02 + t03 + t04 + t05 + t06
# Inputs to the model
input = torch.randn(10, 10)
