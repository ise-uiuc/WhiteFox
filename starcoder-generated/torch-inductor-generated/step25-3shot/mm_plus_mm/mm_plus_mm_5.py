
class Model(torch.nn.Module):
    def forward(self,input1, input2, input3, input4):
        t1 = torch.tanh(input1)
        t2 = torch.sigmoid(input2)
        t3 = torch.sigmoid(input3)
        t4 = t1 * t2
        t5 = t3 * t2
        t6 = t4 - t5
        t7 = t3 + t6
        return t7
# Inputs to the model    
input1 = torch.randn(4, 4)
input2 = torch.randn(4, 4)
input3 = torch.randn(4, 4)
input4 = torch.randn(4, 4)
