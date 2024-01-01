
class Model(torch.nn.Module):
    def forward(self, input, weight):
        torch.addmm(input, input, weight)
        return x
