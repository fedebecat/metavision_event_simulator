from io import StringIO
import os
import numpy as np
import torch
from tqdm import tqdm

from torchvision.utils import make_grid
from metavision_core_ml.utils.torch_ops import normalize_tiles
from metavision_core_ml.video_to_event.video_stream_dataset import make_video_dataset
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_image
from metavision_core_ml.utils.torch_ops import cuda_tick
from profilehooks import profile

import yaml
import cv2
import sys

class SuppressPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

def generate_events(mp4_files, minbs = 1, maxbs = 20):
    # mp4_files = []
    # for dst_path in dst_paths:
    #     for root, _ , files in os.walk(dst_path):
    #         for file in files:
    #             if file.endswith(".mp4") and not file.startswith("event"):
    #                 mp4_files.append(os.path.join(root, file))
    
    #read the hyperparameters from the config file 
    with open("hparams_simulator.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    event_gpu = GPUEventSimulator(config['batch_size'], config['height'], config['width'], config['threshold_mu'],
                                  config['threshold_std'], config['refractory_period'], config['leak_rate_hz'], config['cutoff_hz'], config['shot_noise_hz'])

    event_gpu.to(device) 

    record_video = True
    nrows = 2 ** ((config['batch_size'].bit_length() - 1) // 2)
    padding = '00000'

    for filename in tqdm(mp4_files, desc="Generating events", unit="video"):
        try:
            with SuppressPrints():
                if not os.path.exists(os.path.join(os.path.dirname(filename), "event_frames")):
                    os.mkdir(os.path.join(os.path.dirname(filename), "event_frames"))
                if not os.path.exists(os.path.join(os.path.dirname(filename), "event_video")):
                    os.mkdir(os.path.join(os.path.dirname(filename), "event_video"))
                dl = make_video_dataset(
                    os.path.dirname(filename), config['num_workers'], config['batch_size'], config['height'] , config['width'], config['min_frames_per_video'], config['max_frames_per_video'],
                    min_frames=minbs, max_frames=maxbs, rgb=False)
                dl.to(device)
                pause = False
                last_images = None
                if record_video:
                    pass
                    #out = cv2.VideoWriter(os.path.join(os.path.dirname(filename), 'event_video.mp4'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (config['width'], config['height']))
                for i, batch in enumerate(dl):

                    images = batch['images'].squeeze(dim=0)

                    first_times = batch['first_times']
                    timestamps = batch['timestamps']
                    num_frames = batch['video_len']

                    if pause and last_images is not None:
                        images = last_images


                    # randomize parameters
                    # event_gpu.randomize_broken_pixels(first_times, video_proba=0.1)
                    # event_gpu.randomize_thresholds(first_times)
                    # event_gpu.randomize_cutoff(first_times)
                    # event_gpu.randomize_shot(first_times)
                    # event_gpu.randomize_refractory_periods(first_times)

                    log_images = event_gpu.dynamic_moving_average(images, num_frames, timestamps, first_times) #Converts byte images to lof and performs a pass-band motion blur of incoming images. This simulates the latency of the photodiode w.r.t to incoming light dynamic.

                    mode = config['mode']
                    nbins = config['nbins']
                    if mode == 'counts':
                        idx = 1
                        counts = event_gpu.count_events(log_images, num_frames, timestamps, first_times)
                    elif mode == 'event_volume':
                        voxels = event_gpu.event_volume(log_images, num_frames, timestamps,
                                                        first_times, nbins, mode, split_channels=config['split_channels'])
                        #Computes a volume of discretized images formed after the events, without storing the AER events themselves. We go from simulation directly to this space-time quantized representation.
                        #You can obtain the event-volume of [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by specifying the mode to “bilinear” or you can obtain a stack of histograms if mode is set to “nearest”.
                        if config['split_channels']:
                            counts = voxels[:, nbins:] - voxels[:, :nbins]
                            counts = counts.mean(dim=1)
                        else:
                            counts = voxels.mean(dim=1)
                    else:
                        events = event_gpu.get_events(log_images, num_frames, timestamps, first_times)
                        counts = event_image(events, config['batch_size'], config['height'], config['width'])

                    #print(" events shape: ", events.shape)
                    #convert "events" to image AND PLOTit using plt.imshow
                    #capire questi eventi che cazzo sono!!!

                    im = 255 * normalize_tiles(counts.unsqueeze(1).float(), num_stds=3) #Normalizes tiles, allows us to have normalized views (we filter outliers + standardize)

                    #im = make_grid(im, nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                    im = make_grid(im, nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

                    blur_images = torch.exp(log_images) * 255
                    first_frames_indices = torch.cat((torch.zeros(1, device=images.device), num_frames.cumsum(0)))[
                        :-1].long()
                    imin = make_grid(
                        blur_images[None, ..., first_frames_indices].permute(3, 0, 1, 2), nrow=nrows).detach().cpu().permute(
                        1, 2, 0).numpy().astype(np.uint8)
                    final = np.concatenate((im, imin), axis=1)
                    final = im
                    #save the image
                    index = padding [:-(len(str(i+1)))] + str(i+1)
                    if minbs == 1 and maxbs == 2:
                        if i == 0:
                            pass
                        else:
                            index = padding [:-(len(str(i)))] + str(i)
                            cv2.imwrite(os.path.join(os.path.dirname(filename), "event_frames", f"frame_{index}.jpg"), final)
                    else:
                        cv2.imwrite(os.path.join(os.path.dirname(filename), "event_frames", f"frame_{index}.jpg"), final)
                    if record_video:
                        pass
                        #out.write(final)
                    last_images = images
                #cv2.destroyWindow('all')
                os.system("ffmpeg -r 25 -i " + os.path.join(os.path.dirname(filename), "event_frames", f"frame_%05d.jpg") + " -c:v libx265 -crf 10 -y " + os.path.join(os.path.dirname(filename), "event_video", 'event_video.mp4') + " -loglevel quiet > NUL 2>&1")
                if record_video:
                    pass
                    #out.release()
        except Exception as e:
            print(e)
            continue
        


