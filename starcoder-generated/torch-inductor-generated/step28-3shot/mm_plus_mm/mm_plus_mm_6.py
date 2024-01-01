
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        mm1 = torch.mm(input3, input4)
        mm2 = torch.mm(input1, input2)
        t = mm1 + mm2
        return t.flatten(0, 1)
# Inputs to the model
mm1 = torch.randn(256, 128)
input2 = torch.randn(256, 128)
input3 = torch.randn(256, 128)
input4 = torch.randn(256, 128)
