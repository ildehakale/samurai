import torch
from build_sam import build_sam2_camera_predictor

# 1.1. Modeli oluşturun (config dosyanızın yolunu verin)
model = build_sam2_camera_predictor(
    config_file="configs/sam2.1/sam2.1_hiera_s.yaml",
    ckpt_path="/mnt/data/sam2.1_hiera_small.pt",
    device="cuda",
    mode="eval",
).eval()

# 1.2. Sadece encoder’ı alıyoruz
encoder = model.image_encoder
encoder.eval()

image_size = model.image_size  # örn. 1024 veya 512
dummy_input = torch.randn(1, 3, image_size, image_size, device="cuda")



torch.onnx.export(
    encoder,                        # dönüştüreceğimiz modül
    dummy_input,                    # input örneği
    "sam2_encoder.onnx",            # çıkacak ONNX dosyası
    export_params=True,             # ağırlıkları da içine göm
    opset_version=17,               # opset sürümü (17 veya 18 önerilir)
    do_constant_folding=True,       # constant folding optimizasyonu
    input_names=["input_image"],    # opsiyonel ama okunabilirlik için
    output_names=["image_emb"],     
    dynamic_axes={
        "input_image":  {0: "batch_size"},
        "image_emb":    {0: "batch_size"},
    },
)
print("✅ Encoder ONNX’e döküldü: sam2_encoder.onnx")


