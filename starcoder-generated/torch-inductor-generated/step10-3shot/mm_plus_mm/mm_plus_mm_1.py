
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.matmul(input1, input2)
        t2 = torch.matmul(input3, input4)
        t3 = t1 + t2
        t4 = t3 + input5
        return t4
# Inputs to the model
input1 = torch.randn(20, 20)
input2 = torch.randn(20, 20)
input3 = torch.randn(20, 20)
input4 = torch.randn(20, 20)
input5 = torch.randn(20, 20)
