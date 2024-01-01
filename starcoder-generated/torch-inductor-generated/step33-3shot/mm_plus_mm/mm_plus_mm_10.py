
class Model(torch.nn.Module):
    def forward(self, input1, input2, input3):
        t1 = torch.mm(input1, input1)
        t2 = torch.mm(input2, input3)
        x1 = t1 + t2
        t3 = torch.mm(input2, input3)
        x2 = t1 + t3
        t4 = torch.mm(torch.relu(input1), torch.relu(input2))
        t5 = torch.mm(input2, torch.tanh(input3))
        x3 = t4 + t5
        return x1 + x2 + x3
# Inputs to the model
import torch

input1 = torch.randn(5, 5, requires_grad=True)
input2 = torch.randn(5, 5, requires_grad=True)
input3 = torch.randn(5, 5, requires_grad=True)
