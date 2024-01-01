
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = t1
        for i in range(10):
            t1 = torch.mm(input, input)
            t2 = torch.mm(input, input)
        return t2
# Inputs to the model
input = torch.randn(10, 10)
