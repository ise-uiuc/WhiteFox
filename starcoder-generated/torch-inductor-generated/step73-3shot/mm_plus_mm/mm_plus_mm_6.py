
class Model(torch.nn.Module):
    def forward(self, input1, m):
        h_1 = torch.mm(input1, input1)
        h_2 = torch.mm(input1, m)
        return torch.cat((h_1, h_2), 1)
