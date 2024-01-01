
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input2)
        t4 = torch.mm(input1, input4)
        return t1 + t2 + t3 + t4
# Inputs to the model
input1 = torch.randn(96, 96)
input2 = torch.randn(96, 96)
input3 = torch.randn(96, 96)
input4 = torch.randn(96, 96)
input5 = torch.randn(96, 96)
