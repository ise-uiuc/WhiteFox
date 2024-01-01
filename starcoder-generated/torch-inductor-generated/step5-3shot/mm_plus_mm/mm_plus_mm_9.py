
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(16, 16)
        self.linear2 = torch.nn.Linear(16, 16)
    def forward(self, input1, input2, input3, input4):
        t1 = self.linear1(input1)
        t2 = self.linear1(input2)
        t3 = self.linear2(input3)
        t4 = self.linear1(input4)
        return t1 + t2 + t3 + t4
# Input to the model
input1 = torch.randn(16, 16)
input2 = torch.randn(16, 16)
input3 = torch.randn(16, 16)
input4 = torch.randn(16, 16)
# Model instance
model = Model()
