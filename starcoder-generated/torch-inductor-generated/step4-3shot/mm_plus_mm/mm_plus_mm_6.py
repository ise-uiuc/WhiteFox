
class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input1, input2, input3, input4):
        t1 = input1 * input2
        t2 = input3 * input4
        result = t1 - t2
        return result
# Inputs to the model are not provided
