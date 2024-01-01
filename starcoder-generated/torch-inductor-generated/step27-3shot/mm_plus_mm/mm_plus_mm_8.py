
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
    def forward(self, input1, input2, input3, input4):
        output = self.fc1(input1)
        output = self.fc2(input2)
        output = self.fc3(input3)
        output = self.fc4(input4)
        return output.mm(output.mm(output))
# Input to the model
input1 = torch.randn(20, 5)
input2 = torch.randn(20, 5)
input3 = torch.randn(20, 5)
input4 = torch.randn(20, 5)
