import os
import shutil
import collections
from face_alignment import image_align
import cv2

base_dir = '/mnt/vision-nas/data-sets/celeba/'
img_dir = os.path.join(base_dir, 'img_align_celeba')
save_base_dir = 'D:/Data/celeba'
ld_file = 'list_landmarks_align_celeba.txt'
id_file = 'identity_CelebA.txt'
img_size = 256

# filename and landmarks
ld_fp = open(ld_file, 'r')
ld_lines = ld_fp.readlines()[2:]
fn_list = [x.split(' ')[0] for x in ld_lines]
ld_list = [x.split('.jpg')[-1].rstrip().split('  ') for x in ld_lines]

# identity
id_fp = open(id_file, 'r')
id_lines = id_fp.readlines()
id_list = [x.split(' ')[1].rstrip("\n") for x in id_lines]

# top n-most frequent id
ctr = collections.Counter(id_list)
top_id_list = [x[0] for x in ctr.most_common(3300)]

# data list
data_list = [(fn, ld, id) for fn, ld, id in zip(fn_list, ld_list, id_list) if id in top_id_list]

for i, (fn, ld, id) in enumerate(data_list):
    # 4-point landmarks (left eye, right eye, left mouth, right mouth)
    face_landmarks = [[ld[0], ld[1]],
                      [ld[2], ld[3]],
                      [ld[6], ld[7]],
                      [ld[8], ld[9]]]

    # FFHQ align
    image = cv2.imread(os.path.join(img_dir, fn))
    aligned_image = image_align(image, face_landmarks,
                                output_size=img_size, transform_size=img_size*4,
                                enable_padding=True)

    # save
    save_dir = os.path.join(save_base_dir, 'sort_by_id_aligned', id)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i)), aligned_image)

    print('[{}/{}] save {} for id: {}'.format(i+1, len(data_list), fn, id))
print('done')