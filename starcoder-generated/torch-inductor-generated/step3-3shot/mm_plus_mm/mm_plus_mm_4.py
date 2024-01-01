
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.lin3 = nn.Linear(4, 4)

    def forward(self, input1, input2, input3, input4):
        input1 = torch.matmul(self.lin1(input1), torch.tensor([[[1, 2, 3, 4]]]).float())
        input2 = torch.matmul(input2, torch.tensor([[[2, 3, 1, 4]]]).float())
        input3 = torch.matmul(self.lin2(input3), torch.tensor([[[2, 3, 4, 1]]]).float())
        input4 = torch.matmul(input4, torch.tensor([[[2, 3, 4, 1]]]).float())

        output = torch.add(input1, input2)
        output = torch.add(output, input3)
        output = torch.add(output, input4)

        return output


input1 = torch.rand(3, 4)
input2 = torch.rand(3, 4)
input3 = torch.rand(3, 4)
input4 = torch.rand(3, 4)
