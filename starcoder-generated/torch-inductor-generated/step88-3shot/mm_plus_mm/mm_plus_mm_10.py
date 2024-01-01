
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = input1 * 3
        t2 = torch.bmm(input1.unsqueeze(1), input2.unsqueeze(1)).squeeze(1)
        t3 = torch.bmm(input1.unsqueeze(1), input2.unsqueeze(1)).squeeze(1)
        t2 += t1
        return t1 + t2 + t3

# Inputs to the model
input1 = torch.randn(6, 4, 4)
input2 = torch.randn(4, 4, 6)
input3 = torch.randn(6, 4, 4)
