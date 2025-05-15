import json
import numpy as np
import torch
import clip

import sys
import torch

def drop_text(file, ratio):
    try:
        with open(file, encoding='utf-8') as f:
            dict_list = json.load(f)
    except Exception as e:
        print(f"读取数据时出错: {e}")
        return None

    num_m = len(dict_list)
    num_at = len(dict_list[0]['attributes'])
    rnd = np.random.rand(num_m, num_at)
    concated_attr = []
    for id, t in enumerate(dict_list):
        for k, attr in enumerate(t['attributes']):
            if k<num_at and rnd[id][k] < 0.1 :
                t['attributes'][k] = ''
        str = ','.join(t['attributes'])
        concated_attr.append(str)

    return concated_attr


if __name__ == '__main__':
    filename = sys.argv[1]
    ratio = float(sys.argv[2])

    feature_list = []
    text_list = drop_text(filename, ratio)

    model, preprocess = clip.load('ViT-B-16.pt', device='cuda')
    for txt in text_list:
        text = clip.tokenize(txt).to('cuda')
        with torch.no_grad():
            text_feature = model.encode_text(text)
            feature_list.append(text_feature.cpu().type(torch.float))

    outfile = 'text_feature_' + sys.argv[3] + '.pt'
    torch.save(feature_list, outfile)