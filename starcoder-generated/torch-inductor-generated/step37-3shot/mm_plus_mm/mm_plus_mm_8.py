
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        r1 = torch.matmul(input1, input2) + input3 # The order of arguments is swapped between torch.matmul and add
        r2 = torch.matmul(input3, input2) + input1 # r2 is the same as r1 with input1 and input3 being swapped
        return r1 + r2
# Inputs to the model
input1 = torch.randn(100, 100)
input2 = torch.randn(100, 100)
input3 = torch.randn(100, 100)
