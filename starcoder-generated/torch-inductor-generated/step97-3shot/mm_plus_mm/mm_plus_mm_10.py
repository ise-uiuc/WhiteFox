
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = input1*2 + input2*2
        t2 = input2*3 + input1*4
        t3 = input1*5 + input3*6
        return t1/t2/t3
# Inputs to the model
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
