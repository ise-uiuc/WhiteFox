
class Model(torch.nn.Module):
    def forward(self, input):
        x = torch.mm(input, input) + torch.mm(input, input) # Matrix multiplications are added
        x = torch.mm(input, input)
        t1 = torch.mm(input, input)
        t2 = torch.mm(input, input)
        t3 = torch.mm(input, input)
        t4 = torch.mm(input, input)
        t5 = torch.mm(input, input)
        x = torch.mm(input, input)
        x = torch.mm(input, input)
        x = torch.mm(input, input)
        return torch.mm(input, input) + torch.mm(input, input) # The results of the matrix multiplications are added
# Inputs to the model
input1 = torch.randn(1, 1)
input2 = torch.randn(1, 1)
