
class Model(torch.nn.Module):
    def forward(self, input1):
        # aaa
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input1, input1)        
        t3 = torch.mm(input1, input1)

        s1 = 0
        for i in range(10):
            s1 += i

        t4 = torch.mm(input1, input1)
        t5 = torch.mm(input1, input1)

        s2 = 0
        for i in range(10):
            s2 += i

        t6 = torch.mm(input1, input1)
        t7 = torch.mm(input1, input1)
        t8 = torch.mm(input1, input1)
        t9 = torch.mm(input1, input1)

        s3 = 0
        for i in range(10):
            s3 += i

        t10 = torch.mm(input1, input1)
        t11 = torch.mm(input1, input1)
        t12 = torch.mm(input1, input1)
        t13 = torch.mm(input1, input1)
        t14 = torch.mm(input1, input1)
        return t3
# Inputs to the model
input1 = torch.zeros((1, 1))
