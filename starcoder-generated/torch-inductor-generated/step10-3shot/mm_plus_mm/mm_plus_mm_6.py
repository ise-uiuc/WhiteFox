
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input2)
        return 2 * (t1 + t2)
# Inputs to the model
input1 = torch.randn(3, 3)
input2 = torch.randn(3, 3)
