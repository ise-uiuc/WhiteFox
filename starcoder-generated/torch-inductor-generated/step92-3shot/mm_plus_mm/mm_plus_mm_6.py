
class Model(torch.nn.Module):
    def load_weights(self):
        pass

    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input3, input4)
        t3 = torch.mm(input1, input2) + torch.mm(input3, input4)
        return t3
# Inputs to the model
input1 = torch.randn(64, 64)
input2 = torch.randn(64, 64)
input3 = torch.randn(64, 64)
input4 = torch.randn(64, 64)
