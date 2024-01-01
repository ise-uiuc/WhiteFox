
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.mm(input, input)
        t2 = t1 * 1.1
        t3 = t2.view(1, -1)
        t4 = t3 - t2
        t5 = t4.view(-1)
        return t5.sum()
# Inputs to the model
input = torch.randn(50, 50)
