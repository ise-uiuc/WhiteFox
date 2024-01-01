
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5, input6, input7, input8, input9):
        output = torch.mm(input1, input2)
        output = torch.mm(output, input3)
        output = torch.mm(input4, output)
        output = torch.mm(input5, output)
        output = torch.mm(output, input6)
        output = torch.mm(input7, output)
        output = torch.mm(input8, output)
        return torch.mm(output, input9)
# Inputs to the model
input_1 = torch.rand(4, 5)
input_2 = torch.rand(5, 6)
input_3 = torch.rand(6, 3)
input_4 = torch.rand(3, 4)
input_5 = torch.rand(4, 5)
input_6 = torch.rand(5, 7)
input_7 = torch.rand(7, 2)
input_8 = torch.rand(2, 3)
input_9 = torch.rand(3, 7)
