
class Model(torch.nn.Module):
    def forward(self, model_input):
        t1 = torch.mm(model_input, model_input)
        t2 = torch.mm(model_input, model_input)
        return torch.mm(t1, t2)
# Inputs to the model
model_input = torch.randn(4, 4)
