
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = nn.functional.linear(input1, input2)
        t1 = nn.functional.linear(input1, input4)
        t2 = nn.functional.linear(input1, input2)
        return t1 + t2
# Inputs to the model
input1 = torch.randn(128, 128)
input2 = torch.randn(128, 128)
input3 = torch.randn(128, 128)
input4 = torch.randn(128, 128)
