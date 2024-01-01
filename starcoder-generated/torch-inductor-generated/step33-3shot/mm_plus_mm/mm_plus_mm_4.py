
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.reshape(torch.mm(input, input),[7,21,5,9])
        t2 = torch.reshape(torch.mm(input, input),[21,33,7,3])
        t3 = torch.reshape(torch.mm(input, input),[81,13,4,5])
        t4 = torch.reshape(torch.mm(torch.mm(input, input), input),
                            [20,267,4,4,4])

        return t1 + t2 + t3 + t4
# Inputs to the model
input = torch.randn(482423, 56798)
