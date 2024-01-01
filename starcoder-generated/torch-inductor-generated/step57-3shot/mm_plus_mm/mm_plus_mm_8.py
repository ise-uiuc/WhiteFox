
class Model(torch.nn.Module):
    def forward(self, input1):
        t1 = torch.relu(input1)
        t2 = torch.relu(input1)
        t3 = t1 + t2
        return t3
# Inputs to the model
input1 = torch.randn(3, 3)
