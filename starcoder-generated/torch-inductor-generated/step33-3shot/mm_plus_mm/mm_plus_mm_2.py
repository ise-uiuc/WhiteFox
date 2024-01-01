
class Model(torch.nn.Module):
    def forward(self, model_input):
        x1 = torch.mm(model_input, model_input) # This is a single matrix multiplication
        x1 = torch.mm(model_input, model_input) # This is a single matrix multiplication
        x1 = torch.mm(model_input, model_input) # This is a single matrix multiplication
        x1 = torch.mm(model_input, model_input) # This is a single matrix multiplication
        return x1
# Inputs to the model
model_input = torch.randn(100, 100)
