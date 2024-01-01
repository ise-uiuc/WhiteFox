
class Model(nn.Module):
    def forward(self, input1, input2):
        t1 = torch.mm(input1,input2)
        t2 = torch.mm(input2,input1)
        return torch.sigmoid(0.3*t1+0.5*t2)
# Inputs to the model
input1 = torch.randn(3, 5)
input2 = torch.randn(5, 3)
