
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t = torch.mm(input1, input2)
        a = t + input3
        return a
# Inputs to the model
input1 = torch.randn(200, 200)
input2 = torch.randn(200, 200)
input3 = torch.randn(200, 200)
