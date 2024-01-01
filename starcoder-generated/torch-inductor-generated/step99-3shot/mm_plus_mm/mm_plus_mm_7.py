
class Model(torch.nn.Module):
    def forward(self, input, weight1, weight2, weight3, weight4):
        out0 = torch.mm(input, weight1)
        out1 = torch.mm(weight1, input)
        out2 = torch.mm(input, weight2)
        out3 = torch.mm(weight3, input)
        out = out0 + out1 + out2 + out3
        return out
# Inputs to the model
input = torch.rand(5, 5)
weight1 = torch.rand(5, 5)
weight2 = torch.rand(5, 5)
weight3 = torch.rand(5, 5)
weight4 = torch.rand(5, 5)
