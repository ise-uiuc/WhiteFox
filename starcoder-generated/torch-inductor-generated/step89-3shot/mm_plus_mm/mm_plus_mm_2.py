
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = input1 + input2
        return torch.mm(t1, input1.mm(input2))
# Inputs to the model
input1 = torch.randn(5, 2)
input2 = torch.randn(5, 3)
