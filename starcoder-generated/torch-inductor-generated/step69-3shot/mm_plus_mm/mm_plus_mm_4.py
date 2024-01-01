
class Model(nn.Module):
    def forward(self, input1, input2, input3, input4):
        v1 = torch.mm(input1, input2)
        v2 = torch.mm(input3, input4)
        v2 = torch.mm(input3, input4)
        v2 = torch.mm(input3, input4)
        v2 = torch.mm(input3, input4)
        v2 = torch.mm(input3, input4)
        v2 = torch.mm(input3, input4)
        return v1 + v2
# Inputs to the model
input1 = torch.rand(256, 256)
input2 = torch.rand(256, 256)
input3 = torch.rand(256, 256)
input4 = torch.rand(256, 256)
