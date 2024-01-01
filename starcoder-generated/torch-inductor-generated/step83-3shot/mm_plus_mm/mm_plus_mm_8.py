
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2) # Matrix multiplication
        t2 = torch.mm(input3, input4) # Matrix multiplication
        t3 = t1 + t2 # Addition
        t4 = torch.mm(t3, t3) # 3-fold matrix multiplication
        return t4
# Inputs to the model
input1 = torch.randn(1, 32)
input2 = torch.randn(32, 1)
input3 = torch.randn(1, 48)
input4 = torch.randn(48, 1)
