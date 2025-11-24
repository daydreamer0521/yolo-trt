import sys  # noqa: F401
import os
import struct
import torch


# 文件路径
pt_file = '/home/rc/yolo11trt/det/yolo-det/models/yolo11n.pt'
wts_file = '/home/rc/yolo11trt/det/yolo-det/models/yolo11n.wts'
# 模型类型：'detect', 'cls', 'seg', 'pose', 'obb'
m_type = 'detect'

print(f'Generating .wts for {m_type} model')

# Load model
print(f'Loading {pt_file}')


if not os.path.isfile(pt_file):
    raise SystemExit('你确定文件路径没问题？')
if not wts_file:
    wts_file = os.path.splitext(pt_file)[0] + '.wts'
elif os.path.isdir(wts_file):
    wts_file = os.path.join(
        wts_file,
        os.path.splitext(os.path.basename(pt_file))[0] + '.wts')

# Initialize
device = 'cpu'

# Load model
# model = torch.load(pt_file, map_location=device)  # Load FP32 weights
model = torch.load(pt_file, map_location=device, weights_only=False)
model = model['ema' if model.get('ema') else 'model'].float()

if m_type in ['detect', 'seg', 'pose', 'obb']:
    anchor_grid = model.model[-1].anchors * model.model[-1].stride[..., None, None]

    delattr(model.model[-1], 'anchors')

model.to(device).eval()

with open(wts_file, 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f', float(vv)).hex())
        f.write('\n')
