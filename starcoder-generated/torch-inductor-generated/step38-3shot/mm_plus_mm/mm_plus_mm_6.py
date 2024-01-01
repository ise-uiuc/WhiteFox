
class Model(torch.nn.Module):
    def forward(self, input):
        t1 = torch.matmul(input, input)
        a1 = torch.nn.ReLU(t1)
        a2 = torch.nn.ReLU(t1)
        t2 = torch.matmul(input, input)
        a3 = torch.nn.ReLU(t2)
        a4 = torch.nn.ReLU(t2)
        return a1 + a2 + a3 + a4
# Inputs to the model
input = torch.arange(6).reshape(2,3)
