
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input_1, other, input_2):
        op1 = input_1.flatten(1)
        op2 = torch.add(op1, 4.0)
        op3 = op2.reshape(2, 1, 3)
        op4 = torch.add(op3, other)
        op5 = op4.abs()
        op6 = torch.add(input_2, op5)
        return op6
# Inputs to the model
input_1 = torch.tensor(((3.0, 2.0, 2.0, 1.0), (5.0, 1.0, 1.0, 4.0)), dtype=torch.float)
other = torch.tensor((4.0, 5.0, 3.0, 3.0), dtype=torch.float)
input_2 = torch.tensor(((-1.0, 6.0, -1.0, 6.0), (6.0, -1.0, 6.0, -1.0)), dtype=torch.float)
