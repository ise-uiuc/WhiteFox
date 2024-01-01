
class Model(torch.nn.Module):
    def forward(self, in1, in2, in3, in4, in5, in6, in7, in8, in9):
        t1 = torch.mm(in5, in9) # Matrix multiplication between in5 and in9
        t2 = torch.mm(in1, in2) # Matrix multiplication between in1 and in2
        t3 = torch.mm(in7, in8) # Matrix multiplication between in7 and in8
        t4 = torch.mm(in3, in4) # 
#
# 