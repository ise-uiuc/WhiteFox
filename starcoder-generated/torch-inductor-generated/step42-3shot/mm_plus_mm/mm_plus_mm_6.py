
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        var1 = torch.reshape(input1, (1, 20))
        var1 = var1.view((-1,))
        var2 = input2.view((-1,))
        t1 = torch.mm(var1, var2)
        var3 = input3.view(10, 10)
        var4 = input4.view(10, 10)
        t2 = torch.mm(var3, var4)
        return t1 + t2
# Inputs to the model
input1 = torch.rand([1, 48032])
input2 = torch.rand([1, 128])
input3 = torch.rand([10, 256])
input4 = torch.rand([10, 256])
