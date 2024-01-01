
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input2, input3)
        
        # Replace...
        # t2 = t1 + t2
        #... with the assignment
        if torch.eq(t1, t2):
            t2 = t1
        else:
            t2 = t1 + t2

        t3 = torch.mm(input1, input4)
        t3 = t3 + t1
        
        # Replace...
        # t3 = t3 + t2
        #... with the assignment
        if torch.ne(t1, t3):
            t3 = t3 + t2
        else:
            t3 += t2

        # Replace...
        # t1 = torch.mm(input1, input2)
        #... with the assignment
        if torch.lt(t1, t3):
            t1 = torch.mm(input1, input2)
        else:
            t1 = torch.mm(input1, input4)

        # Replace...
        # t1 = torch.mm(input1, input2)
        #... with the assignment
        if torch.gt(t1, t3):
            t1 = torch.mm(input1, input4)
        else:
            t1 = torch.mm(input1, input2)

        # Replace...
        # t2 = torch.mm(input2, input3)
        #... with the assignment
        if torch.le(t1, t2):
            t2 = torch.mm(input2, input3)
        else:
            t2 = torch.mm(input2, input4)

        # Replace...
        # t2 = torch.mm(input2, input3)
        #... with the assignment
        if torch.ge(t1, t2):
            t2 = torch.mm(input2, input4)
        else:
            t2 = torch.mm(input2, input3)

        # Replace...
        # t3 = torch.mm(input1, input4)
        #... with the assignment
        if torch.eq(t3, t1):
            t3 = torch.mm(input1, input2)
        else:
            t3 = torch.mm(input1, input4)

        # Replace...
        # t3 = torch.mm(input1, input2)
        #... with the assignment
        if torch.ne(t3, t2):
            t3 = t2
        else:
            t3 = torch.mm(input1, input2)

        # Replace...
        # t1 = torch.mm(input1, input4)
        #... with the assignment
        if torch.lt(t3, t1):
            t1 = torch.mm(input1, input4)
        else:
            t1 = torch.mm(input1, input3)

        # Replace...
        # t1 = torch.mm(input1, input3)
        #... with the assignment
        if torch.gt(t3, t1):
            t1 = torch.mm(input1, input3)
        else:
            t1 = torch.mm(input1, input4)

        # Replace...
        # t3 = torch.mm(input1, input3)
        #... with the assignment
        if torch.le(t1, t3):
            t3 = torch.mm(input1, input3)
        else:
            t3 = torch.mm(input1, input4)

        if torch.eq(t1, t3):
            t1 = torch.mm(input1, input2)
        else:
            t1 = torch.mm(input1, input3)

        if torch.eq(t1, input3):
            t1 = torch.mm(input1, input2)
        else:
            t1 = torch.mm(input1, input4)

        if torch.eq(t1, input2):
            t1 = torch.mm(input1, input3)
        else:
            t1 = torch.mm(input1, input4)
        t2 = t1 + t3
        return t2
# Inputs to the model
input1 = torch.randn(8, 8)
input2 = torch.randn(8, 8)
input3 = torch.randn(8, 8)
input4 = torch.randn(8, 8)
