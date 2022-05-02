import random
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac


class Augmenter:
    def __init__(self, aug='all'):

        stopwords = ['[COL]', '[VAL]', 'title', 'name', 'description', 'manufacturer', 'brand', 'specTableContent']

        aug_typo = nac.KeyboardAug(stopwords=stopwords, aug_char_p=0.1, aug_word_p=0.1)
        aug_swap = naw.RandomWordAug(action="swap", stopwords=stopwords, aug_p=0.1)
        aug_del = naw.RandomWordAug(action="delete", stopwords=stopwords, aug_p=0.1)
        aug_crop = naw.RandomWordAug(action="crop", stopwords=stopwords, aug_p=0.1)
        aug_sub = naw.RandomWordAug(action="substitute", stopwords=stopwords, aug_p=0.1)
        aug_split = naw.SplitAug(stopwords=stopwords, aug_p=0.1)

        aug = aug.strip('-')

        augmentations_map = {
            'all': [aug_typo, aug_swap, aug_split, aug_sub, aug_del, aug_crop, None],
            'typo': [aug_typo, None],
            'swap': [aug_swap, None],
            'delete': [aug_del, None],
            'crop': [aug_crop, None],
            'substitute': [aug_sub, None],
            'split': [aug_split, None]
        }
        self.augmentation_array = augmentations_map.get(aug, augmentations_map.get('all'))

    def apply_aug(self, string):
        aug = random.choice(self.augmentation_array)
        if aug is None:
            return string
        else:
            return aug.augment(string)
