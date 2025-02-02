# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from cypath.data.wsi import KFB
import numpy as np
from PIL import Image
import mmcv


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    # test a single image
    max_mag = 40
    curr_mag = 10
    save_mag = 2.5
    psize = 512
    scale = curr_mag / save_mag
    scale = int(scale)
    ssize = psize // scale

    wsi = KFB(args.img, max_mag, curr_mag)
    wsi.setIterator(psize)

    w,h = wsi.getSize()

    M = np.zeros((h//scale+psize, w//scale+psize))
    I = np.zeros((h//scale+psize, w//scale+psize, 3))
    O = np.zeros((h//scale+psize, w//scale+psize, 3))

    x = wsi.x // scale
    y = wsi.y // scale

    for img in wsi:

        img = img[...,::-1]
        result = inference_segmentor(model, img)
        mask = np.array(result).squeeze()
        mask = mask.astype(np.uint8)
        # show the ovlp results
        ovlp = model.show_result(img, result, palette=get_palette(args.palette), opacity=args.opacity)
        ovlp = mmcv.bgr2rgb(ovlp)

        # show in M, I, O
        img = img[...,::-1]
        img = Image.fromarray(img)
        img = img.resize((ssize, ssize))
        img = np.array(img).astype(np.uint8)
        I[y:y+ssize, x:x+ssize, :] = img

        mask = Image.fromarray(mask)
        mask = mask.resize((ssize, ssize))
        mask = np.array(mask).astype(np.uint8)*255
        M[y:y+ssize, x:x+ssize] = mask

        ovlp = Image.fromarray(ovlp)
        ovlp = ovlp.resize((ssize, ssize))
        ovlp = np.array(ovlp).astype(np.uint8)
        O[y:y+ssize, x:x+ssize, :] = ovlp

        x = wsi.x // scale
        y = wsi.y // scale

    M = M[:h//scale, :w//scale]
    M = M.astype(np.uint8)
    Image.fromarray(M).save('/home/yxw1452/Desktop/a.png')
    I = I[:h//scale, :w//scale, :]
    I = I.astype(np.uint8)
    Image.fromarray(I).save('/home/yxw1452/Desktop/a.jpg')
    O = O[:h//scale, :w//scale, :]
    O = O.astype(np.uint8)
    Image.fromarray(O).save('/home/yxw1452/Desktop/a.jpeg')
    return
    show_result_pyplot(
        model,
        args.img,
        result,
        get_palette(args.palette),
        opacity=args.opacity)


if __name__ == '__main__':
    main()
