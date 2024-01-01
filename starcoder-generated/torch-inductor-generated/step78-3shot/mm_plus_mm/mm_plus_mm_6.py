
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        # Matrix Multiplication
        t1 = torch.mm(input1, input1)
        # Addition
        t2 = t1 + input2
        # Matrix Multiplication
        t3 = torch.mm(input1, input1)
        return torch.mm(t1, t2)
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(3, 2)
input3 = torch.randn(2, 3)
