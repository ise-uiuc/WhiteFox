
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t1 = torch.transpose(input1, 1, 2)
        t2 = torch.transpose(input2, 1, 2)
        torch.transpose(input1, 1, 3)
        return torch.transpose(input2, 3, 2)
