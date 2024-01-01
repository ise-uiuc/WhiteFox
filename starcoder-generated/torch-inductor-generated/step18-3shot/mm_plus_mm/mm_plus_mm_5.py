
# This model returns a shape [2, 10] tensor
class Model(torch.nn.Module):
    def forward(self, input1):
        input2 = input1.view(20).unsqueeze_(0).unsqueeze_(0)
        input3 = torch.ones([2, 5])
        output = input2 + input3
        return output
# Inputs to the model
input1 = torch.randn(5, 10)
