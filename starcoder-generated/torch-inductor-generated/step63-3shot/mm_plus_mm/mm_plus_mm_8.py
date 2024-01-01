
class Model(torch.nn.Module):
    def forward(self, input1):
        return torch.mm(input1, input1) + torch.mm(input1, input1) - torch.mm(input1, input1)
# Inputs to the model
input1 = torch.randn(32, 32)
