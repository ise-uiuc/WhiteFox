
class Model(torch.nn.Module):
    def forward(self, input, parameter):
	t1 = torch.mm(input, parameter)
        t1 = torch.mm(input, input)
        return t1
# Inputs to the model
input = torch.randn(7, 7)
parameter = torch.randn(7, 7)
