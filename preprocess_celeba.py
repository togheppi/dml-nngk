import os
import shutil
import collections

base_dir = 'D:/Data/celeba/'
img_dir = os.path.join(base_dir, 'img_align_celeba')
id_file = os.path.join(base_dir, 'identity_CelebA.txt')

fp = open(id_file, 'r')
lines = fp.readlines()
# fn_list = [x.split(' ')[0] for x in lines]
id_list = [x.split(' ')[1].rstrip("\n") for x in lines]

ctr = collections.Counter(id_list)
top_id_list = [x[0] for x in ctr.most_common(3300)]

# top_img_list = [x.split(' ')[0] for x in lines if x.split(' ')[1].rstrip("\n") in top_id_list]

cnt = 0
for i, line in enumerate(lines):
    fn = line.split(' ')[0]
    id = line.split(' ')[1].rstrip("\n")

    if id in top_id_list:
        save_dir = os.path.join(base_dir, 'sort_by_id', id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(os.path.join(img_dir, fn), os.path.join(save_dir, '{}.png'.format(cnt)))
        cnt += 1
        print('[{}/{}] {}th file for id: {}'.format(i+1, len(lines), cnt, id))
print('done')