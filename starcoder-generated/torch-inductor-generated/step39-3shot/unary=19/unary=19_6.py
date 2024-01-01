
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 1)
 
    def forward(self, input):
        v1 = self.linear(input)
        v2 = torch.sigmoid(v1)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
__example_input_array__ = np.array([[1, 2, 3]], dtype=np.float32)
__example_output_array__ = np.array([[0.68010]], dtype=np.float32)
x1 = torch.tensor(__example_input_array__, requires_grad=True)
y1 = m(x1)
y1.backward()
print(x1.grad)
torch.testing.assert_allclose(x1.grad, torch.tensor(__example_output_array__))