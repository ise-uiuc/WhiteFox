
t1 = torch.mm(input1, input2)
t2 = torch.mm(input3, input4)
t3 = torch.mm(input1, input3)
t4 = torch.mm(input1, input4)
t5 = t1 + t2
t6 = t5 - t3 + t4 * 2
