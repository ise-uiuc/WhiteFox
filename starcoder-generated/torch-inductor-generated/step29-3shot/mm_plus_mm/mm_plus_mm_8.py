
class Model(nn.Module):
    def forward(self, input1, input2, input3):
        t1 = nn.Tanh()(input1)
        t2 = input1 + input3
        t3 = nn.PReLU()(-t1)
        t4 = nn.Sigmoid()(input1)
        t5 = nn.Tanh()(input2)
        t6 = nn.Tanh()(input3)
        t7 = input3 + input2
        t8 = nn.ReLU()(input2)
        t9 = nn.ReLU()(t3)
        return t5 + t6 + t7 + t8 + t9
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
