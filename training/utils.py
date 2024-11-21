
import json  # For loading JSON files

def int_keys(str_key_dict) -> dict:
    int_keys_dict = dict()
    for key, val in str_key_dict.items():
        int_keys_dict[int(key)] = val
    return int_keys_dict


def load_dict_from_json(path) -> dict:
    f = open(path, 'r')
    return json.load(f)


def how_many_anchors_n_asb(index_to_cap, n)->int:
    """
    Parameters: index_to_cap is sorted so that 'a_s_b' size is descending. 
    """

    for index, cap in index_to_cap.items():
        if len(cap['a_s_b'].keys()) < n:
            return index

def get_image_id_string(im_id, vg_rel=False, cropped=False) -> str:
    if not vg_rel:
        if str(im_id)[:3]=="bra":
            return f"{str(im_id)}.jpg"
        im_id = int(im_id)
        num_digits = len(str(im_id))
        zero_str = ""
        for _ in range(12 - num_digits):
            zero_str += "0"
        return zero_str + str(im_id) + ".jpg"
    else:
        if not cropped:
            image_str = str(im_id) + '.jpg'
        else:
            image_str = str(im_id) + '_cropped.jpg'
        return image_str