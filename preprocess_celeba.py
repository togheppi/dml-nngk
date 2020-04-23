import os
import shutil
import collections
from face_alignment import image_align
import cv2

base_dir = '/mnt/vision-nas/data-sets/celeba/'
img_dir = os.path.join(base_dir, 'img_align_celeba')
save_base_dir = '../Data/celeba'
id_file = 'identity_CelebA.txt'
ld_file = 'list_landmarks_align_celeba.txt'

fp = open(id_file, 'r')
lines = fp.readlines()
# fn_list = [x.split(' ')[0] for x in lines]
id_list = [x.split(' ')[1].rstrip("\n") for x in lines]

ld_fp = open(ld_file, 'r')
ld_lines = ld_fp.readlines()[2:]


ctr = collections.Counter(id_list)
top_id_list = [x[0] for x in ctr.most_common(3300)]

# top_img_list = [x.split(' ')[0] for x in lines if x.split(' ')[1].rstrip("\n") in top_id_list]

cnt = 0
for i, (line, ld_line) in enumerate(zip(lines, ld_lines)):
    fn = line.split(' ')[0]
    id = line.split(' ')[1].rstrip("\n")
    ld_list = ld_lines[0].split('.jpg')[-1].rstrip().split('  ')
    face_landmarks = [[ld_list[0], ld_list[1]],
                      [ld_list[2], ld_list[3]],
                      [ld_list[6], ld_list[7]],
                      [ld_list[8], ld_list[9]]]

    if id in top_id_list:
        save_dir = os.path.join(save_base_dir, 'sort_by_id_aligned', id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # FFHQ align
        image = cv2.imread(os.path.join(img_dir, fn))
        aligned_image = image_align(image, face_landmarks,
                                    output_size=256, transform_size=1024,
                                    enable_padding=True)

        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(cnt)), aligned_image)
        # shutil.copy(os.path.join(img_dir, fn), os.path.join(save_dir, '{}.png'.format(cnt)))
        cnt += 1
        print('[{}/{}] {}th file for id: {}'.format(i+1, len(lines), cnt, id))
print('done')