from cypath.data.link import linkAllFilesFromSrcToTarL0

tar = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2/train_add_normal'
src1 = '/ssd/Breast/split_L1_10x512_mask_Region9918_n2/train'
src2 = '/raid10/10x512NormalPatchOnInOut'
linkAllFilesFromSrcToTarL0(src1, tar, {})
linkAllFilesFromSrcToTarL0(src2, tar, {})
