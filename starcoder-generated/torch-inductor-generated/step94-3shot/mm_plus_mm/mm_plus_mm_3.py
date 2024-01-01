
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = t1
        for i in range(100):
            t2 = torch.mm(t2, t2)
        return t2
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
