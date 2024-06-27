import argparse
import glob
import os
from tqdm import tqdm
import preprocessing
import generator

def renamer(path):
    file_list = glob.glob(path + "/*.png") + glob.glob(path + "/*.jpg") + glob.glob(path + "/*.jpeg")
    ordered_frames = [glob.glob(path + f'/frame_{index+1}_*')[0] for index in range(len(file_list))]
    print(ordered_frames)
    input("Press Enter to continue...")
    _, extension = os.path.splitext(ordered_frames[0])
    print(extension)
    rel_path = os.path.relpath(path)
    for index, filename in tqdm(enumerate(ordered_frames), desc="Renaming files", total=len(ordered_frames)):
        os.rename(filename, os.path.join(rel_path, f"0000{index}"+extension))


def navigate_all_directories(src_path, quality, dst_path, jpeg_compression, rgb = False):
    # Get all the first level directories in the path
    first_level_directories = [os.path.join(src_path, name) for name in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, name))]
    prefix = "CRF"
    if jpeg_compression:
        prefix = "JPEG"
    if rgb:
        prefix = "RGB_" + prefix
    #go to path parent directory
    parent_path = os.path.dirname(src_path)
    for q in quality:
        dst = os.path.join(parent_path, f"{prefix}{q}")
        if not os.path.exists(dst):
            os.mkdir(dst)
    for directory in tqdm(first_level_directories, desc="Processing directories"):
        basename_directory = os.path.basename(directory)
        for q in quality:
            dst = os.path.join(parent_path, f"{prefix}{q}")
            if not os.path.exists(os.path.join(dst, basename_directory)):
                os.mkdir(os.path.join(dst, basename_directory))
        sub_dir = [os.path.join(directory, name) for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
        for dir in sub_dir:
            basename_dir = os.path.basename(dir)
            for q in quality:
                dst = os.path.join(parent_path, f"{prefix}{q}")
                if not os.path.exists(os.path.join(dst, basename_directory, basename_dir)):
                    os.mkdir(os.path.join(dst, basename_directory, basename_dir))
            frames_dirs = [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
            for frames_dir in frames_dirs:
                has_already_been_preprocessed = True
                basename_frames_dir = os.path.basename(frames_dir)
                dst_paths = []
                for q in quality:
                    dst = os.path.join(parent_path, f"{prefix}{q}")
                    if not os.path.exists(os.path.join(dst, basename_directory, basename_dir, basename_frames_dir)):
                        os.mkdir(os.path.join(dst, basename_directory, basename_dir, basename_frames_dir))
                        has_already_been_preprocessed = False
                    dst_paths.append(os.path.join(dst, basename_directory, basename_dir, basename_frames_dir))
                if not has_already_been_preprocessed:
                    preprocessing.preprocess_frames(frames_dir, dst_paths, "tmp", quality, jpeg_compression, rgb)
        # break
    return [os.path.join(parent_path, f"{prefix}{q}") for q in quality]
               

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rename files in a directory')
    parser.add_argument('path', help='path towards image or video dataset')
    parser.add_argument('--quality', nargs='+', type=int, help="qualities to be used for the output videos.")
    parser.add_argument('--dst-path', type=str , help="destination path of output dataset")
    parser.add_argument('--jpeg-compression', action='store_true', help="compress the images using jpeg compression")
    parser.add_argument('--rgb', action='store_true', help="save only the rgb frames")
    parser.add_argument('--minbs', type=int, help="min batch size", default=1)
    parser.add_argument('--maxbs', type=int, help="max batch size", default=2)
    args = parser.parse_args()
    print(args.quality)
    dst_paths = navigate_all_directories(args.path, args.quality, args.dst_path, args.jpeg_compression, args.rgb)
    print(dst_paths)
    if args.rgb:
        generator.generate_events(dst_paths, args.minbs, args.maxbs)
    # generator.generate_events(dst_paths, args.minbs, args.maxbs)
    #renamer(args.path)