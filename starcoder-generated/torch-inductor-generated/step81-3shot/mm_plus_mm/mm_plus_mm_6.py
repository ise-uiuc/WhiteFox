
A = torch.bmm(input1, input2)
B = torch.bmm(input1, input3)
C = torch.bmm(input2, input4)
D = torch.bmm(input3, input4)
E = torch.bmm(input_3, input_4)
output = torch.bmm(input1, input2) + \
          torch.bmm(input1, input3) + \
          torch.bmm(input2, input4) + \
          torch.bmm(input3, input4)
