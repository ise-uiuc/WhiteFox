
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        a = torch.autograd.Variable(torch.randn(1, 1), requires_grad=True)
        return a+input1+input2
# Inputs to the model
input1 = torch.randn(0, 1)
input2 = torch.randn(1, 1)
