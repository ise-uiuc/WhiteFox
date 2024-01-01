
import torch

class Model(torch.nn.Module):
    def forward(self, model_input):
        t1 = torch.mm(model_input, model_input)
        t2 = torch.mm(model_input, model_input)
        t3 = torch.mm(model_input, model_input)
        t4 = torch.mm(model_input, model_input)
        return t1 + t2 + t3 + t4
# Inputs to the model
model_input = torch.randn(100, 100)
