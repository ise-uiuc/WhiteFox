
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4, input5):
        t1 = torch.sigmoid(input1 + input2)
        t2 =  torch.sigmoid(input3)
        t3 = torch.sigmoid(t1 + t2)
        # Add dropout and return the results of sigmoid()
        return torch.sigmoid(t3 * (input4 + input5))
# Inputs to the model
input1 = torch.randn(3, 5)
input2 = torch.randn(3, 5)
input3 = torch.randn(1, 5)
input4 = torch.randn(3, 5)
input5 = torch.randn(3, 5)
