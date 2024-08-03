# -*- coding: utf-8 -*-
# @Time    : 21/6/9 17:01
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : StoneDataLoader.py

from commons.utils import *
from torch.utils.data import Dataset
from PIL import Image
import random
import torch
from skimage.transform import resize
import nibabel as nib
from skimage import measure
from stone_data import COLOR_DICT
from commons.constant import *
import SimpleITK as sitk
import xml.etree.ElementTree as ET


TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)



def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def open_image_file(filepath):
    if basename(filepath).endswith('.dcm'):
        img = read_dicom(filepath)
        nor_image = normalize_scale(img)
        img = (nor_image * 255).astype(np.uint8)
        size_x, size_y, _ = img.shape  # height, width
    elif basename(filepath).endswith('.png') or basename(filepath).endswith('.jpg'):
        img_pil = Image.open(filepath)
        size_y, size_x = img_pil.size  # width, height
        img = np.asarray(img_pil.convert('RGB'))  # height, width
    else:
        raise Exception('Unknown file extension')
    return img, size_y, size_x


def min_max_voi(mask, superior=10, inferior=10):
    sp = mask.shape
    tp = np.transpose(np.nonzero(mask))
    minx, miny, minz = np.min(tp, axis=0)
    maxx, maxy, maxz = np.max(tp, axis=0)
    minz = 0 if minz - superior < 0 else minz - superior
    maxz = sp[-1] if maxz + inferior >= sp[-1] else maxz + inferior + 1
    miny = 0 if miny - superior < 0 else miny - superior
    maxy = sp[1] if maxy + inferior >= sp[1] else maxy + inferior + 1
    minx = 0 if minx - superior < 0 else minx - superior
    maxx = sp[0] if maxx + inferior >= sp[0] else maxx + inferior + 1
    return minx, maxx, miny, maxy, minz, maxz


def get_one_image_rec(label_path):
    # Image.open(label_path) (w,h,c)
    base_root_dir = label_path.split(basename(label_path))[0]
    txt_file = basename(label_path)[:-4] + '.txt'
    if os.path.exists(join(base_root_dir, txt_file)):
        f2 = open(join(base_root_dir, txt_file), 'r')
        rect_list = []
        for line in f2:
            line_nums = line.split('\n')[0].split(',')
            if len(line_nums) < 5:
                continue
            tmp = []
            for num in line_nums:
                tmp.append(int(num))
            rect_list.append(tmp)
    else:
        label_arr = np.asarray(Image.open(label_path).convert('RGB'))
        # 短边是width，长边是height,label_arr(h,w,c)
        file_name = basename(label_path)[:-4]
        rect_list = []
        for key in COLOR_DICT:
            # print(row[NAME], row['训练组a验证组b'], row['Fistula'],row['序号'])
            label_tmp = label_arr.copy()
            color = COLOR_DICT[key]
            for c in color:
                r_binary = np.where(label_tmp[:, :, 0] == c[0], True, False)
                g_binary = np.where(label_tmp[:, :, 1] == c[1], True, False)
                b_binary = np.where(label_tmp[:, :, 2] == c[2], True, False)
                rgb_binary = r_binary * g_binary * b_binary
                rect_idx = np.where(rgb_binary == True)
                if len(rect_idx[0]) == 0:
                    continue

                contours = measure.label(rgb_binary, connectivity=2, background=0)
                contours_unique = np.unique(contours)[1:]
                ii = 0
                for unique in contours_unique:
                    ii += 1
                    seg_uni = contours.copy()
                    total_pixels = (seg_uni == unique)
                    tp = np.transpose(np.nonzero(total_pixels))
                    x, y = np.min(tp, axis=0)
                    maxx, maxy = np.max(tp, axis=0)
                    h = maxx - x
                    w = maxy - y
                    # label框为线或者点的情况
                    if h == 0 or w == 0:
                        continue
                    label_tmp[x:x + h, y:y + w, 0] = 0
                    label_tmp[x:x + h, y:y + w, 1] = 0
                    label_tmp[x:x + h, y:y + w, 2] = 0

                    rect_list.append([int(key), x, y, maxx, maxy])
        f = open(join(base_root_dir, txt_file), 'w')
        f.write("\n".join(str(item)[1:-1] for item in rect_list))
        f.write("\n")
        f.close()
    return rect_list

def convert_annotation(label_path, class_names):
    in_file = open(label_path)
    tree = ET.parse(in_file)
    root = tree.getroot()
    rect_list = []
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_id = class_names.index(cls)
        xmlbox = obj.find('bndbox')
        rect = [cls_id, int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]
        rect_list.append(rect)
    return rect_list

def get_new_one_image_rec(label_path):
    base_root_dir = label_path.split(basename(label_path))[0]
    txt_file = basename(label_path)[:-4] + '.txt'
    rect_list = []
    if os.path.exists(join(base_root_dir, txt_file)):
        f2 = open(join(base_root_dir, txt_file), 'r')
        for line in f2:
            line_nums = list(map(float, line.split('\n')[0].split()))
            if len(line_nums) < 5:
                continue
            line_nums[0] = int(line_nums[0])
            rect_list.append(line_nums)
    return rect_list

class StoneBasePlkDataSet(Dataset):
    def __init__(self, stone_paths, input_size=(256, 256), augment=False):
        self.input_x = input_size[0]
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.augment = augment
        self.img_final_list = stone_paths

        statis_dict = {}

        for dfs in self.img_final_list:
            cat = int(dfs['image_name'][-7])
            statis_dict[cat] = statis_dict[cat] + 1 if statis_dict.__contains__(cat) else 1

        print('Load images: %d' % (len(self.img_final_list)))
        for key in statis_dict.keys():
            statis_str = 'Color %d, Samples %d' % (
                key, statis_dict[key])
            print(statis_str)

    def __len__(self):
        return int(len(self.img_final_list))

    def __getitem__(self, index):
        return

class StoneBiClsPlkDataSet(StoneBasePlkDataSet):

    def __init__(self, stone_paths, input_size, augment, num_classes=3, stone_label_smoothing=0):
        super(StoneBiClsPlkDataSet, self).__init__(stone_paths, input_size, augment)
        self.num_classes = num_classes
        self.stone_label_smoothing = stone_label_smoothing

        self.img_sameorgan_list = {}  # 存储相同器官位置的idx
        for idx in range(1, 13):
            self.img_sameorgan_list[idx] = []

        self.img_sametype_list = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        for idx in range(len(self.img_final_list)):
            image_text = self.img_final_list[idx]['image_text']
            cat = int(self.img_final_list[idx]['image_name'][-7])

            self.img_sametype_list[cat].append(idx)

        if self.stone_label_smoothing != 0:
            self.stone_probabilities = self.get_probabilities(self.stone_label_smoothing)

        random.seed(777)

    def __getitem__(self, index):
        count = index
        image_name = self.img_final_list[count]['image_name']
        img = self.img_final_list[count]['image_patch']
        rect_list = self.img_final_list[count]['image_label']
        image_text = self.img_final_list[count]['image_text'][:-3]
        cat = int(image_name[-7])

        patch_gt = np.zeros((self.input_x, self.input_y))
        for i in range(len(rect_list)):
            patch_gt[rect_list[i][0]:rect_list[i][2], rect_list[i][1]:rect_list[i][3]] = 1

        cat_pair = 1
        random_idx = cat
        pair_idx = self.img_sametype_list[random_idx][random.randint(0, len(self.img_sametype_list[random_idx]) - 1)]

        image_text_pair = self.img_final_list[pair_idx]['image_text'][:-3]
        img_pair = self.img_final_list[pair_idx]['image_patch']
        pair_rect_list = self.img_final_list[pair_idx]['image_label']

        patch_pair_gt = np.zeros((self.input_x, self.input_y))
        for i in range(len(pair_rect_list)):
            patch_pair_gt[pair_rect_list[i][0]:pair_rect_list[i][2], pair_rect_list[i][1]:pair_rect_list[i][3]] = 1

        if self.augment:
            img = self.augment(np.transpose(img.astype(np.uint8), (1, 2, 0)))
            img_pair = self.augment(np.transpose(img_pair.astype(np.uint8), (1, 2, 0)))

        patch_gt = torch.from_numpy(np.expand_dims(patch_gt, 0))
        patch_pair_gt = torch.from_numpy(np.expand_dims(patch_pair_gt, 0))

        image_patch = img
        label_patch = patch_gt

        if self.num_classes == 3:
            # 尿路结石、非尿路结石
            if cat in [1, 3]:
                cat = 1
            elif cat in [2, 4, 5]:
                cat = 2
        elif self.num_classes == 5:
            # 输尿管结石、静脉石、肾结石、其它
            if cat == 5:
                cat = 4

        organ_idx = np.argmax(image_text[:4])
        if organ_idx == 3:
            myloss_target = torch.eye(self.num_classes)[0]
        else:
            myloss_target = self.stone_probabilities[organ_idx].clone()
            myloss_target[cat] += (1.0 - self.stone_label_smoothing)
        return {
            "image_patch": image_patch,
            "image_cat": cat,
            "image_label": label_patch,
            "image_name": image_name,
            "image_text": image_text,
            "image_patch_pair": img_pair,
            "image_label_pair": patch_pair_gt,
            "image_text_pair": image_text_pair,
            "image_cat_pair": cat_pair,
            "myloss_target": myloss_target,
        }

    def get_Kclass_queue(self):
        random.seed(777)
        q = []
        for image_organ_idx in range(1, len(self.img_sameorgan_list)+1):
            idx = self.img_sameorgan_list[image_organ_idx][
                random.randint(0, len(self.img_sameorgan_list[image_organ_idx]) - 1)]

            img = self.img_final_list[idx]['image_patch']

            if self.augment:
                img = self.augment(np.transpose(img.astype(np.uint8), (1, 2, 0)))

            q.append(torch.unsqueeze(img, dim=0))
        return q

    def getPairInfo(self, pred_class_id):
        pair_idx = self.img_sametype_list[pred_class_id][random.randint(0, len(self.img_sametype_list[pred_class_id]) - 1)]
        image_text_pair = self.img_final_list[pair_idx]['image_text'][:-3]
        img_pair = self.img_final_list[pair_idx]['image_patch']
        if self.augment:
            img_pair = self.augment(np.transpose(img_pair.astype(np.uint8), (1, 2, 0)))
        return img_pair, image_text_pair


    def get_probabilities(self, smoothing):
        stone_probabilities = []
        if self.num_classes == 3:
            stone_distributions = [[113, 158, 13],  # organ0
                                   [104, 209, 0],
                                   [193, 68, 526]]
        elif self.num_classes == 5:
            stone_distributions = [[113, 31, 0, 127, 13],
                                   [104, 40, 0, 169, 0],
                                   [193, 68, 448, 0, 78]]
        elif self.num_classes == 6:
            stone_distributions = [[113, 31, 0, 127, 0, 13],
                                   [104, 40, 0, 169, 0, 0],
                                   [193, 68, 448, 0, 46, 32]]
        else:
            raise ValueError('Exceeds the number of classes')

        for i in range(3):
            stone_distribution = stone_distributions[i]
            cat_distribution = []
            for j in range(self.num_classes):
                cat_distribution.append(smoothing * stone_distribution[j] / sum(stone_distribution))
            stone_probabilities.append(cat_distribution)
        return torch.Tensor(stone_probabilities)


class StoneClsPlkDataSet(StoneBasePlkDataSet):
    def __init__(self, stone_paths, input_size, augment, num_classes=3, stone_label_smoothing=0):
        super(StoneClsPlkDataSet, self).__init__(stone_paths, input_size, augment)
        self.num_classes = num_classes
        random.seed(777)

    def __getitem__(self, index):
        count = index
        image_name = self.img_final_list[count]['image_name']
        image_path = self.img_final_list[count]['image_name']
        img = self.img_final_list[count]['image_patch']
        rect_list = self.img_final_list[count]['image_label']
        image_text = self.img_final_list[count]['image_text']
        cat = int(image_name[-7])

        patch_gt = np.zeros((self.input_x, self.input_y))
        for i in range(len(rect_list)):
            patch_gt[rect_list[i][0]:rect_list[i][2], rect_list[i][1]:rect_list[i][3]] = 1

        if self.augment:
            img = self.augment(np.transpose(img.astype(np.uint8), (1, 2, 0)))

        patch_gt = torch.from_numpy(np.expand_dims(patch_gt, 0))

        image_patch = img
        label_patch = patch_gt

        if self.num_classes == 3:
            # 尿路结石、非尿路结石
            if cat in [1, 3]:
                cat = 1
            elif cat in [2, 4, 5]:
                cat = 2
        elif self.num_classes == 5:
            # 输尿管结石、静脉石、肾结石、其它
            if cat == 5:
                cat = 4


        return {
            "image_patch": image_patch,
            "image_cat": cat,
            "image_label": label_patch,
            "image_name": image_name,
            "image_path": image_path,
            "image_text": image_text
        }

def get_full_image_list(dir='../../../medical_data/Gland/Warwick_QU/patches',
                        post_fix=['.png'], seed=666):
    dir_list = [dir]
    img_list = []
    if len(dir_list) == 0:
        return img_list
    img_filename_list = [sorted(glob(join(dir, '*a.*')))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(basename(img))[0]
        item = [img, ]
        img_path = '{:s}/{:s}'.format(dir, img1_name[:-1] + 'b.png')
        item.append(img_path)
        img_list.append(tuple(item))

    return img_list


def get_full_dicom_list(dir='../../../medical_data/Gland/Warwick_QU/patches',
                        post_fix=['.png'], seed=666):
    dir_list = [dir]
    img_list = []
    if len(dir_list) == 0:
        return img_list
    img_filename_list = [sorted(glob(join(dir, '*.dcm')))]

    for img in img_filename_list[0]:
        if not is_image_file(img):
            continue
        img1_name = os.path.splitext(basename(img))[0]
        item = [img, ]
        img_path = '{:s}/{:s}'.format(dir, img1_name + '_roi.nii.gz')
        item.append(img_path)
        img_list.append(tuple(item))

    return img_list


def read_dicom(scan_path):
    try:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))
        # height width
    except Exception as er:
        print(er)
    img_len = len(image_array.shape)
    if img_len == 4:
        image_array = image_array.squeeze()
    if img_len == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
        image_array = np.tile(image_array, 3)
    return image_array


def load_volume(case_path):
    print(os.path.basename(case_path))
    img1_name = os.path.splitext(basename(case_path))[0]
    vol = nib.load(str(case_path))
    voxel_dim = np.array(vol.header.structarr["pixdim"][1:4], dtype=np.float32)
    return np.transpose(np.squeeze(vol.get_data()), (1, 0))


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from driver import std, mean

    input_x, input_y = 224, 224
    data_path = '../../image_patch/CLAHE_patch_percent6_text_seglabel'
    img_final_list = get_full_image_list(dir=data_path)

    img_final_dfs = []
    img_size = {0: 0, 1: 0, 2: 0}
    for idx in range(len(img_final_list)):
        filepath = img_final_list[idx][0]

        img, _, _ = open_image_file(filepath)
        patch_gt = np.array(Image.open(filepath[:-5] + 'b.png').convert('L'), 'f')
        image_text = np.loadtxt(filepath[:-3] + 'txt', dtype=np.float32)
        cat = int(filepath[-7])

        img = np.transpose(img, (2, 0, 1))  # channel, height, width
        img = resize(img, (img.shape[0], input_x, input_y), order=0, mode='edge',
                     cval=0, clip=True, preserve_range=True, anti_aliasing=False)

        patch_gt = resize(patch_gt, (input_x, input_y), order=3, mode='constant',
                          cval=0, clip=True, preserve_range=True, anti_aliasing=False)
        patch_gt = patch_gt / 255
        rect_list = []
        stone_size = 0
        if cat != 0:
            contours = measure.label(patch_gt, connectivity=2, background=0)
            contours_unique = np.unique(contours)[1:]
            for unique in contours_unique:
                seg_uni = contours.copy()
                total_pixels = (seg_uni == unique)
                tp = np.transpose(np.nonzero(total_pixels))
                x, y = np.min(tp, axis=0)
                maxx, maxy = np.max(tp, axis=0)
                h = maxx - x
                w = maxy - y

                # label框为线或者点的情况
                if h == 0 or w == 0:
                    continue
                rect_list.append([x, y, maxx, maxy])

                if x < 112 < maxx and y < 112 < maxy:
                    area_ratio = h * w / (224 * 224)
                    # if area_ratio > 0.2:
                    if area_ratio > 0.15:
                        stone_size = 2
                    # elif 0.03 < area_ratio <= 0.2:
                    elif 0.03 < area_ratio <= 0.15:
                        stone_size = 1
                    else:
                        stone_size = 0

        if cat != 0:
            img_size[stone_size] += 1
        image_text = np.append(image_text, np.eye(3)[stone_size])
        image_name = basename(filepath)
        image_path = filepath

        # patch_gt = torch.from_numpy(np.expand_dims(patch_gt, 0))

        df = {
            "image_patch": img,
            "image_label": rect_list,
            # "image_label": patch_gt,
            "image_name": image_name,
            "image_text": image_text
        }

        img_final_dfs.append(df)
        print(idx)
    print(img_size)
    torch.save(img_final_dfs, join(data_path, "new_stone_data.pkl"))
