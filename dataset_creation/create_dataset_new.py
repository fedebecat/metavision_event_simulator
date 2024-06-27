import sys
print(sys.path)
# Add /usr/lib/python3/dist-packages/ to PYTHONPATH if the output of print(sys.path) does not mention it.
sys.path.append("/usr/lib/python3/dist-packages/")

import argparse
import glob
import os
from tqdm import tqdm
import preprocessing

from glob import glob
from natsort import natsorted

from io import StringIO
import os
import numpy as np
import torch
from tqdm import tqdm

from torchvision.utils import make_grid
from metavision_core_ml.utils.torch_ops import normalize_tiles
#from metavision_core_ml.video_to_event.video_stream_dataset import make_video_dataset
from metavision_core_ml.video_to_event.video_stream_dataset import StreamDataset, StreamDataLoader, VideoDatasetIterator, pad_collate_fn
from metavision_core_ml.video_to_event.gpu_simulator import GPUEventSimulator
from metavision_core_ml.preprocessing.event_to_tensor_torch import event_image
from metavision_core_ml.utils.torch_ops import cuda_tick
from metavision_core_ml.data.scheduling import Metadata
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

# def generate_events(mp4_files, metadatas, minbs = 1, maxbs = 20):
    
#     #read the hyperparameters from the config file 
#     with open("hparams_simulator.yaml", "r") as file:
#         config = yaml.safe_load(file)
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"

    
#     event_gpu = GPUEventSimulator(config['batch_size'], config['height'], config['width'], config['threshold_mu'],
#                                   config['threshold_std'], config['refractory_period'], config['leak_rate_hz'], config['cutoff_hz'], config['shot_noise_hz'])

#     event_gpu.to(device) 

#     record_video = True
#     nrows = 2 ** ((config['batch_size'].bit_length() - 1) // 2)
#     padding = '00000'

#     for filename in tqdm(mp4_files, desc="Generating events", unit="video"):
#         try:
#             with SuppressPrints():
#                 if not os.path.exists(os.path.join(os.path.dirname(filename), "event_frames")):
#                     os.mkdir(os.path.join(os.path.dirname(filename), "event_frames"))
#                 if not os.path.exists(os.path.join(os.path.dirname(filename), "event_video")):
#                     os.mkdir(os.path.join(os.path.dirname(filename), "event_video"))
#                 dl = make_video_dataset(
#                     os.path.dirname(filename), config['num_workers'], config['batch_size'], config['height'] , config['width'], config['min_frames_per_video'], config['max_frames_per_video'],
#                     metadatas, min_frames=minbs, max_frames=maxbs, rgb=False)
#                 dl.to(device)
#                 pause = False
#                 last_images = None
#                 if record_video:
#                     pass
#                     #out = cv2.VideoWriter(os.path.join(os.path.dirname(filename), 'event_video.mp4'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (config['width'], config['height']))
#                 for i, batch in enumerate(dl):

#                     images = batch['images'].squeeze(dim=0)

#                     first_times = batch['first_times']
#                     timestamps = batch['timestamps']
#                     num_frames = batch['video_len']

#                     if pause and last_images is not None:
#                         images = last_images


#                     # randomize parameters
#                     # event_gpu.randomize_broken_pixels(first_times, video_proba=0.1)
#                     # event_gpu.randomize_thresholds(first_times)
#                     # event_gpu.randomize_cutoff(first_times)
#                     # event_gpu.randomize_shot(first_times)
#                     # event_gpu.randomize_refractory_periods(first_times)

#                     log_images = event_gpu.dynamic_moving_average(images, num_frames, timestamps, first_times) #Converts byte images to lof and performs a pass-band motion blur of incoming images. This simulates the latency of the photodiode w.r.t to incoming light dynamic.

#                     mode = config['mode']
#                     nbins = config['nbins']
#                     if mode == 'counts':
#                         idx = 1
#                         counts = event_gpu.count_events(log_images, num_frames, timestamps, first_times)
#                     elif mode == 'event_volume':
#                         voxels = event_gpu.event_volume(log_images, num_frames, timestamps,
#                                                         first_times, nbins, mode, split_channels=config['split_channels'])
#                         #Computes a volume of discretized images formed after the events, without storing the AER events themselves. We go from simulation directly to this space-time quantized representation.
#                         #You can obtain the event-volume of [Unsupervised Event-based Learning of Optical Flow, Zhu et al. 2018] by specifying the mode to “bilinear” or you can obtain a stack of histograms if mode is set to “nearest”.
#                         if config['split_channels']:
#                             counts = voxels[:, nbins:] - voxels[:, :nbins]
#                             counts = counts.mean(dim=1)
#                         else:
#                             counts = voxels.mean(dim=1)
#                     else:
#                         events = event_gpu.get_events(log_images, num_frames, timestamps, first_times)
#                         counts = event_image(events, config['batch_size'], config['height'], config['width'])

#                     #print(" events shape: ", events.shape)
#                     #convert "events" to image AND PLOTit using plt.imshow
#                     #capire questi eventi che cazzo sono!!!

#                     im = 255 * normalize_tiles(counts.unsqueeze(1).float(), num_stds=3) #Normalizes tiles, allows us to have normalized views (we filter outliers + standardize)

#                     #im = make_grid(im, nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)
#                     im = make_grid(im, nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)

#                     blur_images = torch.exp(log_images) * 255
#                     first_frames_indices = torch.cat((torch.zeros(1, device=images.device), num_frames.cumsum(0)))[
#                         :-1].long()
#                     imin = make_grid(
#                         blur_images[None, ..., first_frames_indices].permute(3, 0, 1, 2), nrow=nrows).detach().cpu().permute(
#                         1, 2, 0).numpy().astype(np.uint8)
#                     final = np.concatenate((im, imin), axis=1)
#                     final = im
#                     #save the image
#                     index = padding [:-(len(str(i+1)))] + str(i+1)
#                     if minbs == 1 and maxbs == 2:
#                         if i == 0:
#                             pass
#                         else:
#                             index = padding [:-(len(str(i)))] + str(i)
#                             cv2.imwrite(os.path.join(os.path.dirname(filename), "event_frames", f"frame_{index}.jpg"), final)
#                     else:
#                         cv2.imwrite(os.path.join(os.path.dirname(filename), "event_frames", f"frame_{index}.jpg"), final)
#                     if record_video:
#                         pass
#                         #out.write(final)
#                     last_images = images
#                 #cv2.destroyWindow('all')
#                 os.system("ffmpeg -r 25 -i " + os.path.join(os.path.dirname(filename), "event_frames", f"frame_%05d.jpg") + " -c:v libx265 -crf 10 -y " + os.path.join(os.path.dirname(filename), "event_video", 'event_video.mp4') + " -loglevel quiet > NUL 2>&1")
#                 if record_video:
#                     pass
#                     #out.release()
#         except Exception as e:
#             print(e)
#             continue
        

def make_video_dataset(
        path, num_workers, batch_size, height, width, min_length, max_length, metadatas, mode='frames',
        min_frames=5, max_frames=30, min_delta_t=5000, max_delta_t=50000, rgb=False, seed=None, batch_times=1,
        pause_probability=0.5, max_optical_flow_threshold=2., max_interp_consecutive_frames=20,
        max_number_of_batches_to_produce=None, crop_image=False):
    """
    Makes a video / moving picture dataset.

    Args:
        path (str): folder to dataset
        batch_size (int): number of video clips / batch
        height (int): height
        width (int): width
        min_length (int): min length of video
        max_length (int): max length of video
        mode (str): 'frames' or 'delta_t'
        min_frames (int): minimum number of frames per batch
        max_frames (int): maximum number of frames per batch
        min_delta_t (int): in microseconds, minimum duration per batch
        max_delta_t (int): in microseconds, maximum duration per batch
        rgb (bool): retrieve frames in rgb
        seed (int): seed for randomness
        batch_times (int): number of time steps in training sequence
        pause_probability (float): probability to add a pause during the sequence (works only with PlanarMotionStream)
        max_optical_flow_threshold (float): maximum allowed optical flow between two consecutive frames (works only with PlanarMotionStream)
        max_interp_consecutive_frames (int): maximum number of interpolated frames between two consecutive frames (works only with PlanarMotionStream)
        max_number_of_batches_to_produce (int): maximum number of batches to produce. Makes sure the stream will not
                                                produce more than this number of consecutive batches using the same
                                                image or video.
    """
    #metadatas = build_metadata(path, min_length, max_length)
    print('scheduled streams: ', len(metadatas))

    def iterator_fun(metadata):
        return VideoDatasetIterator(
            metadata, height, width, rgb=rgb, mode=mode, min_tbins=min_frames,
            max_tbins=max_frames, min_dt=min_delta_t, max_dt=max_delta_t, batch_times=batch_times,
            pause_probability=pause_probability, max_optical_flow_threshold=max_optical_flow_threshold,
            max_interp_consecutive_frames=max_interp_consecutive_frames, crop_image=crop_image)
    dataset = StreamDataset(metadatas, iterator_fun, batch_size, "data", None, seed)
    dataloader = StreamDataLoader(dataset, num_workers, pad_collate_fn)
    # TODO: one day unify make_video_dataset and make_video_dataset_with_events_cpu
    dic_params_video_dataset = {"height": height, "width": width,
                                "min_tbins": min_frames, "max_tbins": max_frames,
                                "rgb": rgb, "pause_probability": pause_probability,
                                "max_number_of_batches_to_produce": max_number_of_batches_to_produce}

    dataloader.dic_params_video_dataset = dic_params_video_dataset
    return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files in a directory')
    #parser.add_argument('path', help='path towards image or video dataset')
    parser.add_argument('--quality', nargs='+', type=int, help="qualities to be used for the output videos.")
    #parser.add_argument('--dst-path', type=str , help="destination path of output dataset")
    parser.add_argument('--jpeg-compression', action='store_true', help="compress the images using jpeg compression")
    parser.add_argument('--rgb', action='store_true', help="save only the rgb frames")
    parser.add_argument('--minbs', type=int, help="min batch size", default=1)
    parser.add_argument('--maxbs', type=int, help="max batch size", default=2)
    args = parser.parse_args()
    print(args.quality)
    #dst_paths = navigate_all_directories(args.path, args.quality, args.dst_path, args.jpeg_compression, args.rgb)
    #print(dst_paths)
    #if args.rgb:
    #   generator.generate_events(dst_paths, args.minbs, args.maxbs)
    # generator.generate_events(dst_paths, args.minbs, args.maxbs)
    #renamer(args.path)

    #src_paths = ['test/test.avi', 'test/test2.avi']
    #src_paths = glob(f'/media/becattini/399B724D60527D8A/workspace/metavision_event_simulator/src_path/JPG_10/*/*/*/video.mkv')
    src_paths = glob('/media/becattini/399B724D60527D8A/workspace/metavision_event_simulator/test/*.avi')
    print(src_paths)

    metadatas = []
    for path in src_paths:
        metadatas.append(Metadata(path, 0, 30)) # assume each video has 30 fps
#    generate_events(src_paths, metadatas, args.minbs, args.maxbs)

    minbs = args.minbs
    maxbs = args.maxbs
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

    for filename in tqdm(src_paths, desc="Generating events", unit="video"):
        # try:
        #     with SuppressPrints():
        if not os.path.exists(os.path.join(os.path.dirname(filename), "event_frames")):
            os.mkdir(os.path.join(os.path.dirname(filename), "event_frames"))
        if not os.path.exists(os.path.join(os.path.dirname(filename), "event_video")):
            os.mkdir(os.path.join(os.path.dirname(filename), "event_video"))
        dl = make_video_dataset(
            os.path.dirname(filename), config['num_workers'], config['batch_size'], config['height'] , config['width'], config['min_frames_per_video'], config['max_frames_per_video'],
            metadatas, min_frames=minbs, max_frames=maxbs, rgb=False)
        dl.to(device)
        pause = False
        last_images = None
        if record_video:
            pass
            #out = cv2.VideoWriter(os.path.join(os.path.dirname(filename), 'event_video.mp4'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (config['width'], config['height']))
        for i, batch in enumerate(dl):

            images = batch['images'].squeeze(dim=0)

            first_times = batch['first_times']
            print('first_times', first_times)
            timestamps = batch['timestamps']
            print('timestamps', timestamps)
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
        # except Exception as e:
        #     print(e)
        #     continue