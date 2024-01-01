
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(torch.mm(input1, input2), input3)
        return torch.mm(t1, input4)
# Inputs to the model
input1 = torch.randn(2, 2) # 2-D tensor to test vectorized matrix multiplication and broadcasting rules
input2 = torch.randn(2, 2)
input3 = torch.randn(2, 2)
input4 = torch.randn(2, 2)
