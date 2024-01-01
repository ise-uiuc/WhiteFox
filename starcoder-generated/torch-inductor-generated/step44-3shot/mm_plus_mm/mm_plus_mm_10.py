
class TorchScriptModel(torch.nn.Module):
    def __init__(self):
        super(T, self).__init__()

        def T(input):
            t1 = torch.mm(input, input)
            t2 = torch.mm(input, input)
            t3 = torch.mm(input, input)
            t4 = torch.mm(input, input)
            t5 = t1 + t2 + t3 + t4
            t6 = torch.mm(input, input)
            t7 = t5 + t6
            t8 = torch.mm(input, input)
            t9 = t6 + t8
            t10 = t7 + t9
            return t10
        self.model_ft = T
# Inputs to the model
x1 = torch.randn(4, 4)
