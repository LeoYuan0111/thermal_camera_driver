import sys
sys.path.append('../plantProj')

import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from cameras.wrapper_boson import BosonWithTelemetry

def parse_args():
    parser = argparse.ArgumentParser(description="Record thermal video from two Boson cameras.")
    parser.add_argument('--output', type=str, help='Output file name.', required=True)
    parser.add_argument('--duration', type=int, default=-1, help='Duration of the recording in seconds. Default is -1 (record until manually stopped).')
    return parser.parse_args()

def main():
    args = parse_args()

    output_path = "./dual_data/" / Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file = str(output_path)

    def shutdown_handler():
        if 'thr_cam_obj_A' in locals():
            thr_cam_obj_A.close()
        if 'thr_cam_obj_B' in locals():
            thr_cam_obj_B.close()
        print("Cameras closed.")
        sys.exit()

    try:
        thr_cam_obj_A = BosonWithTelemetry(device=1, port="COM4")
    except:
        print("Failed to connect to thermal camera A.")
        shutdown_handler()
    try:
        thr_cam_obj_B = BosonWithTelemetry(device=2, port="COM6")
    except:
        print("Failed to connect to thermal camera B.")
        shutdown_handler()
    
    thr_cam_obj_A.camera.do_ffc()
    thr_cam_obj_B.camera.do_ffc()
    thr_cam_obj_A.camera.set_ffc_manual()
    thr_cam_obj_B.camera.set_ffc_manual()
    time.sleep(1)
    print("Cameras connected, FFC performed and set to manual.")

    start_recording = input("Press Enter to start recording...")
    if start_recording.lower() == 'q':
        print("Recording cancelled.")
        shutdown_handler()
    
    try:
        start_time = time.time()
        thr_cam_obj_A.start_logging()
        thr_cam_obj_B.start_logging()
        duration = args.duration if args.duration > 0 else float('inf')
        if duration == float('inf'):
            print("Recording started. Press 'q' to stop.")
        else:
            print(f"Recording started for {args.duration} seconds. Press 'q' to stop early.")
        fig = plt.figure()
        while time.time() - start_time < duration:
            if plt.waitforbuttonpress(timeout=0.1):
                if not duration == float('inf'):
                    print("Recording stopped early by user.")
                break
            time.sleep(0.1)
        
        thr_cam_obj_A.stop_logging()
        thr_cam_obj_B.stop_logging()
        print("Recording finished.")

        raw_thr_frames_A = np.array(thr_cam_obj_A.logged_images)
        raw_thr_tstamps_A = np.array(thr_cam_obj_A.logged_tstamps)
        thr_cam_timestamp_offset_A = thr_cam_obj_A.timestamp_offset

        raw_thr_frames_B = np.array(thr_cam_obj_B.logged_images)
        raw_thr_tstamps_B = np.array(thr_cam_obj_B.logged_tstamps)
        thr_cam_timestamp_offset_B = thr_cam_obj_B.timestamp_offset


        np.savez(output_file, raw_thr_frames_A=raw_thr_frames_A, raw_thr_tstamps_A=raw_thr_tstamps_A, thr_cam_timestamp_offset_A=thr_cam_timestamp_offset_A,
                raw_thr_frames_B=raw_thr_frames_B, raw_thr_tstamps_B=raw_thr_tstamps_B, thr_cam_timestamp_offset_B=thr_cam_timestamp_offset_B)
        print(f"Data saved to {output_file}")
    except KeyboardInterrupt:
        print("Saving interrupted by user.")
    finally:
        thr_cam_obj_A.close()
        thr_cam_obj_B.close()


if __name__ == "__main__":
    main()
