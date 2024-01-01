
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 1024
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        block2 = Bottleneck(64, 64, downsample=nn.Sequential(nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False)))
        layer3 = self._make_layer(block2, 256, layers[0])
        self.layer3 = nn.Sequential(*layer3)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False)
        block4 = Bottleneck(1024, 1024, downsample=nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)))
        layer5_2 = Bottleneck(512, 512, downsample=nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)))
        layer6_1 = Bottleneck(256, 256)
        layer6_2 = Bottleneck(256, 256)
        layer6_3 = Bottleneck(256, 256)
        layer6_4 = Bottleneck(256, 256)
        layer7 = [layer5_2, layer6_1, layer6_2, layer6_3, layer6_4]
        self.layer7 = nn.Sequential(*layer7)
        self.conv4 = nn.Conv2d(512, 2048, kernel_size=1, stride=1, bias=False)
        self.conv5 = nn.Conv2d(2048, 2048, kernel_size=1, stride=1, bias=False)
        block8 = Bottleneck(2048, 2048, downsample=nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1, stride=1, bias=False)))
        layer9 = self._make_layer(block8, 1024, layers[3])
        self.layer9 = nn.Sequential(*layer9)
        self.conv6 = nn.ConvTranspose2d(1024, 512, kernel_size=1, stride=1, bias=False)
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv7 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        layer11 = Bottleneck(512, 512)
        layer12 = Bottleneck(512, 512)
        layer13 = Bottleneck(512, 512)
        self.layer13 = nn.Sequential(*([layer11, layer12, layer13]))
        self.conv11 = nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1, bias=False)
        self.upSample_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv12 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv13 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv14 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        layer15 = Bottleneck(256, 256)
        layer16 = Bottleneck(256, 256)
        layer17 = Bottleneck(256, 256)
        self.layer17 = nn.Sequential(*([layer15, layer16, layer17]))
        self.conv15 = nn.ConvTranspose2d(256, 128, kernel_size=1, stride=1, bias=False)
        self.upSample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv16 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv17 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        layer18 = Bottleneck(128, 128)
        layer19 = Bottleneck(128, 128)
        self.layer19 = nn.Sequential(*([layer18, layer19]))
        self.conv18 = nn.ConvTranspose2d(128, 64, kernel_size=1, stride=1, bias=False)
        self.upSample_3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv19 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False)
        layer20 = Bottleneck(64, 64)
        layer21 = Bottleneck(64, 64)
        self.layer22 = nn.Sequential(*([layer20, layer21]))
        self.conv22 = nn.ConvTranspose2d(64, 32, kernel_size=1, stride=1, bias=False)
        self.upSample_4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv23 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        layer24 = Bottleneck(32, 32)
        self.layer25 = nn.Sequential(*([layer24]))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = self.maxpool(v2)
        v4 = self.layer3(v3)
        v5 = self.conv2(v4)
        v6 = self.conv3(v5)
        v7 = v2 + v6
        v8 = self.conv4(v7)
        v9 = self.conv5(v8)
        v10 = v1 + v9
        v11 = self.layer7(v10)
        v12 = self.conv4(v11)
        v13 = self.conv5(v12)
        v14 = v9 + v13
        v15 = self.layer9(v14)
        v16 = self.conv6(v15)
        v17 = self.upSample(v16)
        v18 = self.conv7(v11)
        v19 = v18 + v17
        v20 = torch.relu(v19)
        v21 = self.conv8(v20)
        v22 = torch.relu(v21)
        v23 = self.conv9(v22)
        v24 = torch.relu(v23)
        v25 = self.conv10(v24)
        v26 = torch.relu(v25)
        v27 = self.layer13(v26)
        v28 = self.conv11(v27)
        v29 = self.upSample_1(v28)
        v30 = self.conv12(v27)
        v31 = v30 + v29
        v32 = torch.relu(v31)
        v33 = self.conv13(v32)
        v34 = torch.relu(v33)
        v35 = self.conv14(v34)
        v36 = torch.relu(v35)
        v37 = self.layer17(v36)
        v38 = self.conv15(v37)
        v39 = self.upSample_2(v38)
        v40 = self.conv16(v37)
        v41 = v40 + v39
        v42 = torch.relu(v41)
        v43 = self.conv17(v42)
        v44 = torch.relu(v43)
        v45 = self.layer19(v44)
        v46 = self.conv18(v45)
        v47 = self.upSample_3(v46)
        v48 = self.conv19(v45)
        v49 = v48 + v47
        v50 = torch.relu(v49)
        v51 = self.layer22(v50)
        v52 = self.conv22(v51)
        v53 = self.upSample_4(v52)
        v54 = self.conv23(v51)
        v55 = v54 + v53
        v56 = torch.relu(v55)
        v57 = self.layer25(v56)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride!= 1 or self.inplanes!= planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride), nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        layer = nn.Sequential(*layers)
        return layer

# Inputs to the model
x1 = torch.randn(1, 3, 512, 512)
torch.onnx.export(Model(), x1, model_file_name)

# model-arch-description-end

# Description of the input tensor. Please provide the shape, type and size here. 
{
    "input_shape": None, 
    "input_type": "torch.Tensor", 
    "input_size(MB)": None
}

# Sample test code for PyTorch. Please make sure your test code covers all necessary cases. 
import torch
import torch.onnx.operators
import torch.nn as nn
import onnx
import onnxruntime
import json
import pytest

def check_model(*init_args, **run_args):
  model = Model(*init_args)
  model.eval()

  # Input to the model
  x = torch.randn(1, 3, 128, 128, device='cpu')

  torch.onnx.export(model, x, model_file_name, verbose=True)

  # Verify with onnxruntime
  session = onnxruntime.InferenceSession(model_file_name)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  x = x.detach().numpy()
  got = session.run([output_name], {input_name: x})[0]
  np.testing.assert_allclose(model(torch.from_numpy(x)).detach().numpy(), got, rtol=1e-03, atol=1e-05)

check_model()
