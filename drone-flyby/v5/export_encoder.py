"""Export the same pretrained DINOv2, then measure actual ONNX CPU execution."""
import time
import argparse
import os
import resource
from pathlib import Path
import torch
import numpy as np
import onnxruntime as ort
from v5.common import require_azure, save_json, timing


class EncoderOnly(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, images):
        return self.encoder.forward_features(images)['x_norm_clstoken']


def main():
    require_azure()
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', type=int, nargs='+', default=[98,140,224])
    parser.add_argument('--batches', type=int, nargs='+', default=[8,16,32,64])
    parser.add_argument('--repeats', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    assets = Path('/assets')
    model = torch.hub.load(str(assets/'dinov2'), 'dinov2_vits14', source='local', pretrained=False).eval()
    model.load_state_dict(torch.load(assets/'dinov2_vits14.pth', map_location='cpu', weights_only=True))
    model = EncoderOnly(model).eval()
    rows = []
    for size in args.sizes:
        path = assets/f'dinov2_{size}.onnx'
        x = torch.randn(1,3,size,size)
        torch.onnx.export(model, x, str(path), input_names=['images'], output_names=['features'],
            dynamic_axes={'images':{0:'batch'}, 'features':{0:'batch'}}, opset_version=17, dynamo=False)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), sess_options=opts, providers=['CPUExecutionProvider'])
        for batch in args.batches:
            values = []
            inp = np.repeat(x.numpy(), batch, axis=0)
            cpu_start = time.process_time()
            wall_start = time.perf_counter()
            for i in range(args.repeats+2):
                start = time.perf_counter()
                output = session.run(None, {'images': inp})[0]
                if i > 1:
                    values.append((time.perf_counter()-start)*1000)
            cpu_percent = 100*(time.process_time()-cpu_start)/(time.perf_counter()-wall_start)
            with torch.inference_mode():
                expected = model(torch.from_numpy(inp[:1])).numpy()
            error = float(np.max(np.abs(output[:1]-expected)))
            assert error < 0.005, error
            rows.append({'encoder':'dinov2_vits14_onnx', 'size':size, 'batch':batch,
                         'max_abs_export_error':error, 'cpu_percent':cpu_percent,
                         'process_peak_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         'cuda_available':torch.cuda.is_available(), 'cuda_count':torch.cuda.device_count(),
                         'repeats':args.repeats, **timing(values)})
            save_json('/results/encoder_capacity.json', rows)
    print(rows, flush=True)


if __name__ == '__main__':
    main()
