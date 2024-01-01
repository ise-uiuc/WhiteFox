
class Model(torch.nn.Module):
    def forward(self, input1, input2):
        t = torch.mm(input1, input2)
        return t + t
# Inputs to the model
input1 = torch.randn(5, 5).unsqueeze(0)
input2 = torch.randn(5, 5).unsqueeze(0)
