#!/usr/bin/env python3
import argparse
import time
import csv

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast
import torch.nn.functional as F
import onnxruntime as ort

# 1) Torch-TensorRT’i try/except ile içe aktarın
try:
    import torch_tensorrt as trtorch
    TRT_AVAILABLE = True
except ImportError:
    print("[WARN] torch-tensorrt import edilemedi, TRT derlemesi yapılamayacak.")
    TRT_AVAILABLE = False

from sam2.build_sam import build_sam2_camera_predictor

# ONNX wrapper (önceden tanımladığınız)
class ONNXEncoderWrapper(torch.nn.Module):
    def __init__(self, onnx_path, prompt_encoder, mask_decoder, emb_size, img_size, device="cuda"):
        super().__init__()
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, sess_options=sess_opts,
            providers=[p for p in ["TensorrtExecutionProvider","CUDAExecutionProvider","CPUExecutionProvider"]
                       if p in ort.get_available_providers()]
        )
        self.input_name     = self.sess.get_inputs()[0].name
        self.output_name    = self.sess.get_outputs()[0].name
        self.prompt_encoder = prompt_encoder
        self.mask_decoder   = mask_decoder
        self.embedding_size = emb_size
        self.image_size     = img_size
        self.device         = device
        # MaskDecoder içinden conv katmanları alıyoruz
        self.conv_s0 = mask_decoder.conv_s0
        self.conv_s1 = mask_decoder.conv_s1

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x_np = x.cpu().numpy().astype(np.float32)
        emb = self.sess.run([self.output_name], {self.input_name: x_np})[0]
        emb_t = torch.from_numpy(emb).to(self.device)
        emb_t = F.interpolate(emb_t, size=(self.embedding_size, self.embedding_size),
                              mode="bilinear", align_corners=False)
        # Projeksiyon
        f0 = self.conv_s0(emb_t)
        f1 = self.conv_s1(emb_t)
        f2 = F.interpolate(f0, scale_factor=0.5,  mode="bilinear", align_corners=False)
        f3 = F.interpolate(f0, scale_factor=0.25, mode="bilinear", align_corners=False)
        # PosEnc
        pe = self.prompt_encoder.get_dense_pe().to(self.device)
        pe_list = [pe, pe, pe, pe]
        return {"backbone_fpn":[f0,f1,f2,f3],"vision_pos_enc":pe_list}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",    required=True)
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--source",    default="0")
    p.add_argument("--csv",       default="")
    p.add_argument("--color",     type=int, nargs=3, default=[0,255,0])
    return p.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device=="cuda" else torch.float32

    cudnn.benchmark = True
    cudnn.deterministic = False

    # -------------------------------------------------------------------
    # 2) Predictor’u oluşturun (sabit)
    predictor = build_sam2_camera_predictor(
        config_file=args.config,
        ckpt_path=args.checkpoint,
        device=device,
        mode="eval"
    )

    # -------------------------------------------------------------------
    # 3) image_encoder’i Torch-TensorRT ile derleyin (tam modül)
    if TRT_AVAILABLE and device=="cuda":
        try:
            trt_enc = trtorch.compile(
                predictor.image_encoder,
                inputs=[trtorch.Input((1,3,predictor.image_size,predictor.image_size),
                                      dtype=torch.float32)],
                enabled_precisions={torch.float, torch.half}
            )
            predictor.image_encoder = trt_enc
            print("[INFO] Başarıyla TRT motoru elde edildi.")
        except AssertionError as e:
            print(f"[WARN] TRT derlemesi başarısız: {e}\n → ONNX wrapper’a dönülüyor.")
            predictor.image_encoder = ONNXEncoderWrapper(
                onnx_path=args.onnx,
                prompt_encoder=predictor.sam_prompt_encoder,
                mask_decoder=predictor.sam_mask_decoder,
                emb_size=predictor.sam_image_embedding_size,
                img_size=predictor.image_size,
                device=device
            )
    else:
        predictor.image_encoder = ONNXEncoderWrapper(
            onnx_path=args.onnx,
            prompt_encoder=predictor.sam_prompt_encoder,
            mask_decoder=predictor.sam_mask_decoder,
            emb_size=predictor.sam_image_embedding_size,
            img_size=predictor.image_size,
            device=device
        )

    # -------------------------------------------------------------------
    # 4) Decoder katmanlarını FP16’a alın
    predictor.sam_mask_decoder.half()
    predictor.sam_prompt_encoder.half()

    # -------------------------------------------------------------------
    # 5) Kaynak açma, ROI seçme, load_first_frame vs. (orijinal akış)
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Kaynak açılamadı.")
    bbox = cv2.selectROI("ROI Seçin", frame, showCrosshair=True)
    cv2.destroyAllWindows()
    x,y,w,h = bbox
    first_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    with autocast(device_type=device, dtype=dtype):
        predictor.load_first_frame(first_rgb)
    with autocast(device_type=device, dtype=dtype):
        _, obj_ids, mask_logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=1,
            bbox=(x, y, x + w, y + h)
        )

    # -------------------------------------------------------------------
    # 6) Takip döngüsü ve görselleştirme
    prev_time = time.time()
    loop_start = prev_time


    frame_count = 1

    csv_file = open(args.csv, "w", newline="") if args.csv else None
    csv_writer = csv.writer(csv_file) if csv_file else None

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with autocast(device_type=device, dtype=dtype):
            obj_ids, mask_logits = predictor.track(frame_rgb)

        vis = frame.copy()
        found_any = False
        for logit in mask_logits:
            mask = (logit[0] > 0.0).cpu().numpy()
            ys, xs = np.where(mask)
            if ys.size > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max),
                              tuple(args.color), 2, cv2.LINE_AA)
                found_any = True

        # Eğer bbox bulunamadıysa dilerseniz boş bir kutu veya hiç kutu çizmeyebilirsiniz.
        # Örneğin hiç bulunamazsa kırmızı bir uyarı çiz:
        if not found_any:
            cv2.putText(vis, "No object!", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        curr_time = time.time()
        fps_inst = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(vis, f"FPS: {fps_inst:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow("KILIC TRT Demo", vis)
        if cv2.waitKey(1) & 0xFF == 27: break

        if csv_writer:
            t = time.time() - loop_start
            fps = frame_count / t if t>0 else 0.0
            csv_writer.writerow([frame_count, fps])


    cap.release()
    cv2.destroyAllWindows()
    if csv_file: csv_file.close()

    total_t = time.time() - loop_start
    print(f"Ortalama FPS: {frame_count/total_t:.2f}")

if __name__=="__main__":
    main()
