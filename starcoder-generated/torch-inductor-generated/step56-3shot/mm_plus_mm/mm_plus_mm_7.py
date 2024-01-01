
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = input1 + torch.mm(input2, torch.mm(input1, input2))
        return 3.0 + torch.mm(t1, t1)*0.5
# Inputs to the model
input1 = torch.randn(298, 298)*3
input2 = torch.randn(298, 298)*2
