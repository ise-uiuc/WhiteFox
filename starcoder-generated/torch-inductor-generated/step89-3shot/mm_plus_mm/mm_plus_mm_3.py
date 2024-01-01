
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mv(input1, input2)
        t2 = torch.mv(input2, input3)
        t3 = torch.mv(input1, input3)
        return t1 * t2 / t3
# Inputs to the model
input 1 = torch.randn(6, 6)
input 2 = torch.randn(6, 1)
input 3 = torch.randn(6, 1)
