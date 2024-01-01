
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mv(input1, input2) / input3
        return input2.mv(input3.mv(t1))
# Inputs to the model
input1 = torch.randn(5, 5)
input2 = torch.randn(5, 5)
input3 = torch.randn(5, 5)
