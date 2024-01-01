
class Model1(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.matmul(input1, input2)
        t2 = torch.matmul(input3, input2)
        t3 = t1 + t2
        return t3
class Model2(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input2)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(64, 64)
input2 = torch.randn(64, 64)
input3 = torch.randn(64, 64)
