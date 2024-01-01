
class Model(torch.nn.Module):
    def forward(self, in1):
        return torch.mm(in1, in1) + 2.0
# Inputs to the model
in1 = torch.randn(64, 64)
