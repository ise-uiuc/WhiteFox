
class Model(torch.nn.Module):
    def forward(self, input, input5):
        t1 = torch.mm(input, input5)
        t2 = torch.mm(input5, input)
        t3 = t1 + t2
        return t3
# Inputs to the model
input = torch.randn(5, 5)
input5 = torch.randn(5, 3)
