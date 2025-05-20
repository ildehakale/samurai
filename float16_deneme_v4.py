import argparse
import torch
import numpy as np
import cv2
import torch.backends.cudnn as cudnn
from torch.amp import autocast
from sam2.build_sam import build_sam2_camera_predictor
import time
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive live demo for SAMURAI real-time segmentation (FP16 + cuDNN)"
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
    parser.add_argument(
        "--bbox",
        type=int,
        nargs=4,
        metavar=("X", "Y", "W", "H"),
        help="Provide bounding box coordinates as x y w h (disables ROI selection)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to save tracking results as CSV"
    )
    return parser.parse_args()

def move_module_to_fp16(obj, device):
    from torch import nn
    if isinstance(obj, nn.Module):
        obj.to(device).half()
    elif isinstance(obj, dict):
        for v in obj.values():
            move_module_to_fp16(v, device)
    else:
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if isinstance(child, nn.Module) or isinstance(child, dict):
                move_module_to_fp16(child, device)

def main():
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

    predictor = build_sam2_camera_predictor(args.config, args.checkpoint, device=device, mode="eval")

    # Only move to FP16 and compile if it's a torch.nn.Module
    if isinstance(predictor, torch.nn.Module):
        move_module_to_fp16(predictor, device)
        predictor = torch.compile(predictor)
        predictor.eval()

    width, height = 640, 480
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"Error: cannot open source {source}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: cannot read first frame from source")
        return
    frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

    if hasattr(predictor, "load_first_frame"):
        with autocast(enabled=True, dtype=torch.float16, device_type=device_type):
            predictor.load_first_frame(frame_rgb)

    if args.bbox:
        x, y, w, h = args.bbox
        print(f"[INFO] Using bbox from arguments: x={x}, y={y}, w={w}, h={h}")
        if w <= 0 or h <= 0:
            print("Invalid bbox dimensions. Exiting.")
            return
    else:
        bbox = cv2.selectROI("Select Object to Track", first_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Object to Track")
        x, y, w, h = bbox
        if w == 0 or h == 0:
            print("No bounding box selected, exiting.")
            return

    bbox_pts = np.array([[x, y], [x + w, y + h]], dtype=np.float32)

    if hasattr(predictor, "add_new_prompt"):
        with autocast(enabled=True, dtype=torch.float16, device_type=device_type):
            _, obj_ids, mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=1, bbox=bbox_pts
            )

    row_idx = 0
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, mode='w', newline='')
        csv_writer = csv.writer(csv_file)

    box_color = tuple(args.color)
    prev_time = time.time()
    loop_start = prev_time
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if hasattr(predictor, "track"):
            with autocast(enabled=True, dtype=torch.float16, device_type=device_type):
                obj_ids, mask_logits = predictor.track(frame_rgb)
        else:
            continue

        vis = frame.copy()
        found_any = False

        for logit in mask_logits:
            mask = (logit[0] > 0.0).cpu().numpy()
            ys, xs = np.where(mask)
            if ys.size > 0:
                x_min, x_max = int(xs.min()), int(xs.max())
                y_min, y_max = int(ys.min()), int(ys.max())
                cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), box_color, 2, cv2.LINE_AA)
                if csv_writer:
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min
                    row_idx += 1
                    csv_writer.writerow([row_idx, x_min, y_min, bbox_width, bbox_height])
                    found_any = True

        if csv_writer and not found_any:
            row_idx += 1
            csv_writer.writerow([row_idx, 0, 0, 0, 0])

        curr_time = time.time()
        fps_inst = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(vis, f"FPS: {fps_inst:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("KILIC Tracker", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and not args.bbox:
            if hasattr(predictor, "reset_state"):
                predictor.reset_state()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, first_frame2 = cap.read()
            if not ret2:
                break
            frame_rgb2 = cv2.cvtColor(first_frame2, cv2.COLOR_BGR2RGB)
            if hasattr(predictor, "load_first_frame"):
                with autocast(enabled=True, dtype=torch.float16, device_type=device_type):
                    predictor.load_first_frame(frame_rgb2)
                    predictor.add_new_prompt(
                        frame_idx=0, obj_id=1, bbox=np.array([[x, y], [x + w, y + h]], dtype=np.float32)
                    )

    if csv_file:
        csv_file.close()

    total_time = time.time() - loop_start
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

