
class A(torch.nn.Module):
    def __init__(self, A):
        super().__init__()
        self.A = A

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module1 = A(torch.nn.Linear(1, 1))
        self.module2 = A(self.module1)
    def forward(self, input_tenosr):
        res = self.module1(input_tenosr)
        a1 = res.permute(0, 2, 1)
        return a1
model = Model()
model.eval()

x1 = torch.randn(1, 1, 1)

script = torch.jit.script(model)
input_names = ["x1"]
output_names = ["output:0"] 
y= script(x1) # Call the model with input
outputs = [(y1,y2) for y1, y2 in outputs] 
print(outputs)
torch.jit.save(script, "model.pt") # Export the JIT Model to the ONNX Model


