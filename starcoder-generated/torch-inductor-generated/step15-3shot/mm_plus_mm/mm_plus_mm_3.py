
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = input1 + input1
        t2 = input1 + input1
        t3 = t1 + t2
        t4 = t1 + t2
        t5 = t3 + t4
        return t5
# Inputs to the model
input1 = torch.randn(100, 100)
