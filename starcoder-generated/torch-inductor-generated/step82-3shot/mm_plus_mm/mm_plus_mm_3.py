
class Model1(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.nn.functional.linear(input1, input2)
        t2 = torch.nn.functional.linear(input3, input4)
        t3 = t1 + t2
        return torch.relu(t3)
class Model2(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.matmul(input1, input2)
        t2 = torch.matmul(input3, input4)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(6, 6)
input2 = torch.randn(6, 6)
input3 = torch.randn(6, 6)
input4 = torch.randn(6, 6)

input1 = torch.randn(64, 64)
input2 = torch.randn(64, 64)
input3 = torch.randn(64, 64)
input4 = torch.randn(64, 64)
