
class Model(torch.nn.Module):
    def forward(self, data1):
        data1 = torch.mm(data1, data1)
        return data1
# Inputs to the model
data1 = torch.randn(128,64)
