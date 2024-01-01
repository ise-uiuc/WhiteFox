
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        for i in range(10):
            t2 = torch.mm(input, input)
            t1 = t1 + t2
        return t1
# Inputs to the model
input = torch.randn(10, 10)
