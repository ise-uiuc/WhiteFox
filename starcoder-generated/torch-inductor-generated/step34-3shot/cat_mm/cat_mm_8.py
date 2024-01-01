
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        list1 = []
        list1.append(x)
        list1.append(x)
        list2 = []
        for loopVar2 in range(5):
            list2.append(list1)
        list3 = list2 + list2
        return list3[0]
# Input to the model
x = torch.randn(3)
