
class Model(torch.nn.Module):
    def forward(self, model_input):
        v1 = torch.mm(model_input, model_input)
        return v1
# Inputs to the model
model_input = torch.randn(5, 5)
