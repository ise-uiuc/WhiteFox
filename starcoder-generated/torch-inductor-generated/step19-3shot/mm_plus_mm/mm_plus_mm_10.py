
class Model(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()


class Block(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input3)
        t2 = torch.mm(input2, input4)
        t3 = torch.mm(t1, t2)
        t4 = torch.randn(5, 5)
        return t4


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for _ in range(2):
            name = "model" + str(_)
            layer = Block()
            self.add_module(name, layer)
    def forward(self, x):
        v1 = self.model0(x, x, x, x)
        v2 = self.model1(v1, v1, v1, v1)
        return v2
# Inputs to the model
x = torch.randn(5, 5)
