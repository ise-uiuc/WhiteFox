
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t = torch.reshape(input3, (56, 2, 3, 4))
        t = t.transpose(1,0)
        t = torch.reshape(t, (96, 12))
        return torch.mm(t, input2)
# Inputs to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(56, 8)
