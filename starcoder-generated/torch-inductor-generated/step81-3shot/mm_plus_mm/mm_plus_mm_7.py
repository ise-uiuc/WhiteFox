
class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
        def forward(self, inputf, input1, input2, input3, input4):
            t1 = torch.mm(input1, input2)
            t2 = torch.mm(input2, input1)
            t4 = torch.mm(input3, input1)
            t5 = torch.mm(input4, input4)
            t3 = t1 + t1 + t2 + t4 + t5
            return t3
        # Inputs to the model
        inputf = torch.rand(4, 10)
        input1 = torch.rand(4, 10)
        input2 = torch.rand(4, 10)
        input3 = torch.rand(10, 20)
        input4 = torch.rand(10, 20)
        