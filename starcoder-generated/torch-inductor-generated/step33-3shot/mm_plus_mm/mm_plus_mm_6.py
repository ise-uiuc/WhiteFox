
class Model(torch.nn.Module):
    def forward(self, model_input):
        x = torch.mm(model_input, model_input)
        x = torch.mm(model_input, model_input)
        x = x + x
        y = x + x
        z = torch.mm(model_input, model_input)
        return y + z
# Inputs to the model
model_input = torch.randn(6, 6)
