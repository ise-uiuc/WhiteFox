
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input5, input1)
        return t1 + t2 + t3
# Inputs to the model
input1 = torch.randn(1, 12)
input2 = torch.randn(12,8)
input3 = torch.randn(11, 12)
input4 = torch.randn(126, 128)
input5 = torch.randn(128, 1)
# Model input ends
