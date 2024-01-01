
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, torch.transpose(input, 1, 0))
        return t1 + t2
# Inputs to the model
input = torch.randn(5, 5)
