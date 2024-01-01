
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.mm(input1, input1) # Matrix multiplication between input1 and input1
        t2 = t1 + t1 # Addition of the results of matrix multiplications (t1 & t2)
        t3 = torch.mm(t2, t2) # Matrix multiplication of the results of the addition of t1 & t2 and t1 & t2
        return t1 + t3 # Addition of the results of matrix multiplications t1 & t3 and matrix multiplication of the results of the addition t1 & t2 & t1 & t3
# Inputs to the model
input1 = torch.randn(20, 20)
