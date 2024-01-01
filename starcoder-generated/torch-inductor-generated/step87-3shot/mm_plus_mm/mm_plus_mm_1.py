
class simple_model(torch.nn.Module):
    def __init__(self):
        super(simple_model, self).__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = t1 + t2 # Addition of the results of the two matrix multiplications
        return t3
sm = simple_model()
# Inputs to the model
input1 = torch.randn(10, 10)
input2 = torch.randn(10, 10)
input3 = torch.randn(10, 10)
input4 = torch.randn(10, 10)
