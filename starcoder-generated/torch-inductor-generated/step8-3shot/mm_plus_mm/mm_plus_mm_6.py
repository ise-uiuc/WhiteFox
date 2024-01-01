
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3, input4):
        t1 = torch.mm(input1, input2)
        t2 = torch.mm(input1, input4)
        t1_relu = TorchF.relu(t1)
        t2_relu = F.relu(t2)
        t3 = torch.mm(input3, input2)
        t4 = torch.mm(input3, input4)
        t3_relu = F.relu(t3)
        t4_relu = torch.mm(input4, t4)
        t5 = t1_relu + t2_relu
        t6 = t3_relu + t4_relu
        return t5 * t6
# Inputs to the model
input1 = torch.randn(2, 3)
input2 = torch.randn(3, 5)
input3 = torch.randn(2, 3)
input4 = torch.randn(3, 5)
