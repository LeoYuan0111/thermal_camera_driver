import argparse
import time
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

from wrapper_boson import BosonWithTelemetry

def parse_args():
    parser = argparse.ArgumentParser(description="Record thermal video from a Boson camera.")
    parser.add_argument('--output', type=str, help='Output file path.', required=True)
    parser.add_argument('--duration', type=int, default=10, help='Duration of the recording in seconds.')
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)


    try:
        thr_cam_obj = BosonWithTelemetry()
    except:
        print("Failed to connect to thermal camera")
        sys.exit()
    thr_cam_obj.camera.do_ffc()
    time.sleep(1)

    print("Camera connected and FFC performed.")

    start_recording = input("Press Enter to start recording...")
    if start_recording.lower() == 'q':
        print("Recording cancelled.")
        sys.exit()
    
    start_time = time.time()
    thr_cam_obj.start_logging()
    print("Recording started. Press 'q' to stop early.")
    while time.time() - start_time < args.duration:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped early by user.")
            break
        time.sleep(0.1)
    
    thr_cam_obj.stop_logging()
    print("Recording finished.")

    raw_thr_frames = np.array(thr_cam_obj.logged_images)
    raw_thr_tstamps = np.array(thr_cam_obj.logged_tstamps)
    thr_cam_timestamp_offset = thr_cam_obj.timestamp_offset

    np.savez(output_file, raw_thr_frames=raw_thr_frames, raw_thr_tstamps=raw_thr_tstamps, thr_cam_timestamp_offset=thr_cam_timestamp_offset)
    print(f"Data saved to {output_file}")

    thr_cam_obj.close()


if __name__ == "__main__":
    main()
