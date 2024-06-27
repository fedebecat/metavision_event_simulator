#open the frames in order
#center crop the frames
#make a video
#save it in the dst path

import glob

import cv2
import os
import shutil
import platform


def center_crop(img):
    #i need a square image
    #i need to find the smallest dimension
    #i need to crop the image to the smallest dimension
    #i need to find the center of the image
    #i need to crop the image to the center
    height, width, _ = img.shape
    smallest_dimension = min(height, width)
    start_row = (height - smallest_dimension) // 2
    start_col = (width - smallest_dimension) // 2
    end_row = start_row + smallest_dimension
    end_col = start_col + smallest_dimension
    return img[start_row:end_row, start_col:end_col]

def preprocess_frames(src_path, dst_paths, tmp_path, qualities, jpeg_compression, rgb):
    assert len(dst_paths) == len(qualities)
    file_list = glob.glob(src_path + "/*.jpg") + glob.glob(src_path + "/*.jpeg")
    def split_and_sort(filename):
        if platform.system() == "Windows":
            file = filename.split('\\')[-1]
        else:
            file = filename.split('/')[-1]
        parts = file.split('_')
        return int(parts[1])

    file_list.sort(key = split_and_sort)
    padding ='00000'
    tmp = os.path.join(os.getcwd(), tmp_path)
    if not os.path.exists(tmp):
        os.mkdir(tmp)
    if jpeg_compression:
        for q in qualities:
            if not os.path.exists(os.path.join(tmp, f"Q{q}")):
                os.mkdir(os.path.join(tmp, f"Q{q}"))
        for i, filename in enumerate(file_list):
            index = padding [:-(len(str(i+1)))] + str(i+1)
            file_new_name = f"frame_{index}.jpg"
            #open the images (filname)
            #center crop the images
            try:
                img = cv2.imread(filename)
                img = center_crop(img)
                #save the images
                for q in qualities:
                    cv2.imwrite(os.path.join(tmp, f"Q{q}", file_new_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            except Exception as e:
                print(e)
                continue
        for dst_path, quality in zip(dst_paths, qualities):
            os.system("ffmpeg -r 25 -i " + os.path.join(tmp,f"Q{quality}") + f"/frame_%05d.jpg -c:v libx265 -crf 10 -y " + dst_path + f"/output{quality}.mp4 -loglevel quiet > NUL 2>&1")
            if rgb:
                fname = os.path.join(dst_path, f"output{quality}.mp4")
                command = f"ffmpeg -i {os.path.join(fname)} -vf fps={25} {dst_path}/frame_%05d.jpg -crf 0 -q:v 2 -loglevel quiet > NUL 2>&1"
                os.system(command)
                os.remove(os.path.join(dst_path, f"output{quality}.mp4"))
    else:
        for i, filename in enumerate(file_list):
            index = padding [:-(len(str(i+1)))] + str(i+1)
            file_new_name = f"frame_{index}.jpg"
            #open the images (filname)
            #center crop the images
            try:
                img = cv2.imread(filename)
                img = center_crop(img)
                #save the images
                cv2.imwrite(os.path.join(tmp, file_new_name), img)
            except Exception as e:
                print(e)
                continue
        for dst_path, quality in zip(dst_paths, qualities):
            os.system("ffmpeg -r 25 -i " + tmp + f"/frame_%05d.jpg -c:v libx265 -crf {quality} -y " + dst_path + f"/output{quality}.mp4 -loglevel quiet > NUL 2>&1")
            if rgb:
                fname = os.path.join(dst_path, f"output{quality}.mp4")
                command = f"ffmpeg -i {os.path.join(fname)} -vf fps={25} {dst_path}/frame_%05d.jpg -crf 0 -q:v 2 -loglevel quiet > NUL 2>&1"
                os.system(command)
                os.remove(os.path.join(dst_path, f"output{quality}.mp4"))
        #delele the tmp folder
    try:
        shutil.rmtree(tmp)
    except OSError as e:
        print("Error deleting folder: %s : %s" % (tmp, e.strerror))
    
def preprocess_rgb_frames(src_path, dst_paths, qualities, jpeg_compression):
    assert len(dst_paths) == len(qualities)
    file_list = glob.glob(src_path + "/*.jpg") + glob.glob(src_path + "/*.jpeg")
    def split_and_sort(filename):
        file = filename.split('\\')[-1]
        parts = file.split('_')
        return int(parts[1])

    file_list.sort(key = split_and_sort)
    padding ='00000'
    if jpeg_compression:
        for i, filename in enumerate(file_list):
            index = padding [:-(len(str(i+1)))] + str(i+1)
            file_new_name = f"frame_{index}.jpg"
            #open the images (filname)
            #center crop the images
            try:
                img = cv2.imread(filename)
                img = center_crop(img)
                #save the images
                for dst_path, q in zip(dst_paths, qualities):
                    cv2.imwrite(os.path.join(dst_path, file_new_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            except Exception as e:
                print(e)
                continue
    else:
        for i, filename in enumerate(file_list):
            index = padding [:-(len(str(i+1)))] + str(i+1)
            l_path = filename.split('\\')[-4:-1]
            file_new_name = f"frame_{index}.jpg"
            #open the images (filname)
            #center crop the images
            try:
                img = cv2.imread(filename)
                img = center_crop(img)
                #save the images
                cv2.imwrite(os.path.join(dst_paths, file_new_name), img)
            except Exception as e:
                print(e)
                continue

if __name__ == "__main__":
    preprocess_frames(r'C:\Users\super\Downloads\facemorphic_annotated_rgb\recordings\ad1de3a2297b49c79359257e96caa770\AU_1\frames_1', [r"C:\Users\super\Downloads\facemorphic_annotated_rgb\Q50_", r"C:\Users\super\Downloads\facemorphic_annotated_rgb\Q25_", r"C:\Users\super\Downloads\facemorphic_annotated_rgb\Q10_"], "tmp", [50, 25, 10])
