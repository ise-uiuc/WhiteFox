
class Model(torch.nn.Module):
    def forward(self, model_input):
        v1 = torch.mm(model_input, torch.rand(3, 7))
        v2 = torch.nn.functional.relu(v1)
        v3 = v1 + v2
        return v3
# Inputs to the model
model_input = torch.randn(30, 30)
