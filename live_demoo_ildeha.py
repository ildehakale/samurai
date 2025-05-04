
import argparse
import torch
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from torch.amp import autocast
from sam2.build_sam import build_sam2_camera_predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive live demo for SAM2 real-time segmentation (FP16 + cuDNN)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_s.yaml",
        help="Path to the model config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam2.1_hiera_small.pt",
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (0 for webcam or path to video file)"
    )
    parser.add_argument(
        "--color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        default=(0, 255, 0),  # default box color: green
        help="BGR color for the bounding box"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    # Build predictor
    predictor = build_sam2_camera_predictor(args.config, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    # cuDNN extension
    cudnn.enabled = True
    cudnn.benchmark = True           # find the best conv algorithm for your input shapes
    cudnn.deterministic = False       # allow non-deterministic but faster algorithms
    cudnn.allow_tf32 = True          # allow TensorFloat-32 on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True

    # Recursively find & convert any sub-module of predictor to FP16
    def move_module_to_fp16(obj):
        from torch import nn
        if isinstance(obj, nn.Module):
            obj.to(device).half()
        elif isinstance(obj, dict):
            for v in obj.values():
                move_module_to_fp16(v)
        else:
            # walk through attributes
            for attr in dir(obj):
                if attr.startswith("_"):
                    continue
                try:
                    child = getattr(obj, attr)
                except Exception:
                    continue
                if isinstance(child, nn.Module) or isinstance(child, dict):
                    move_module_to_fp16(child)

    move_module_to_fp16(predictor)
    predictor.eval()


    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: cannot open source {source}")
        return

    # Read first frame and select ROI
    ret, first_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame from source")
        return

    bbox = cv2.selectROI("Select Object to Track", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Object to Track")
    x, y, w, h = bbox
    if w == 0 or h == 0:
        print("No bounding box selected, exiting.")
        return
    bbox_pts = np.array([[x, y], [x + w, y + h]], dtype=np.float32)

    # Load first frame into predictor under autocast
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    with autocast(enabled=True, dtype=torch.float16,device_type=device_type):
        predictor.load_first_frame(frame_rgb)

    # Add the prompt (initial bbox)
    with autocast(enabled=True, dtype=torch.float16,device_type= device_type):
        _, obj_ids, mask_logits = predictor.add_new_prompt(
            frame_idx=0, obj_id=1, bbox=bbox_pts
        )

    box_color = tuple(args.color)

    # Tracking loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Inference in FP16
        with autocast(enabled=True, dtype=torch.float16,device_type = device_type):
            obj_ids, mask_logits = predictor.track(frame_rgb)

        vis = frame.copy()
        # Draw bounding boxes from mask logits
        for logit in mask_logits:
            mask = (logit[0] > 0.0).cpu().numpy()
            ys, xs = np.where(mask)
            if ys.size > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                cv2.rectangle(
                    vis,
                    (x_min, y_min),
                    (x_max, y_max),
                    box_color,
                    thickness=2,
                    lineType=cv2.LINE_AA
                )

        cv2.imshow("SAMURAI Tracker", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset and re-select ROI
            predictor.reset_state()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, first_frame2 = cap.read()
            if not ret2:
                break
            bbox2 = cv2.selectROI("Select Object to Track", first_frame2, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select Object to Track")
            x2, y2, w2, h2 = bbox2
            if w2 == 0 or h2 == 0:
                break

            bbox_pts2 = np.array([[x2, y2], [x2 + w2, y2 + h2]], dtype=np.float32)
            frame_rgb2 = cv2.cvtColor(first_frame2, cv2.COLOR_BGR2RGB)
            with autocast(enabled=True, dtype=torch.float16,device_type = device_type):
                predictor.load_first_frame(frame_rgb2)
                _, obj_ids, mask_logits = predictor.add_new_prompt(
                    frame_idx=0, obj_id=1, bbox=bbox_pts2
                )

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
