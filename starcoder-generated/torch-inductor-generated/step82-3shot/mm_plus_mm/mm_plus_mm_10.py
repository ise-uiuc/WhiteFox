
class Model(torch.nn.Module):
    def forward(self, input):
        return torch.mm(input.transpose(0, 1), input.transpose(0, 1))
# Inputs to the model
input = torch.randn(64, 64)
