from typing import Tuple

import pandas as pd
from pandas import DataFrame
from torch.utils.data import Dataset

from src.predictors.contrastive import ContrastiveDataCollator, ContrastivePredictor
from src.preprocess.configs import ExperimentsArgumentParser


class TurkLabeledDataset(Dataset):

    def __init__(self, pretrain_df: DataFrame):
        self.pretrain_df = pretrain_df

        self.targets = self.pretrain_df['target_id'].unique()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index) -> DataFrame:
        """
        :param index:
        :return: a data frame with the fields:
            - target_id - the id of the fyndiq item the offer was sampled for (can be the fyndiq offer itself)
            - cluster_id - id of the cluster to which the offer belongs
            - text - the actual name of the offer
            - source - '#1' of the offer comes from fyndiq, '#2' otherwise
        """
        target_id = self.targets[index]

        return self.pretrain_df[self.pretrain_df['target_id'] == target_id]


class ContrastivePretrainingTurksDatasetCollator(ContrastiveDataCollator):
    """
    TODO:
    The dataset above will return all the offers that were gathered for one target product.

    Using that I need to generate pairs of matching offers (or duplicates, but the idea is that it has to be
    pairs for the SupCon loss function)

    What I am thinking:
    - the sett up "batch size" would be half (or some other ratio) of the actual targeted batch. For explanation purposes
    let's say that the target batch size is 128, so we set it to 64
    - This means that 64 dataframes will arrive at the collator. From each dataset we can select one matching combination
    and one un-matching combination (or more such combinations if we want to triple the size in this step instead of
    making it double.
    - Nevertheless, we need to take care that not more than one non-matching example is taken from each dataset that
    reached this point. As such we make sure that no potential matches in-between 2 items from google shopping are
    treated as non matches by the algorithm

    Each target product is paired with 10 potential matching offers (it may be less in cases where google shopping
    returned fewer products. There is no guarantee that any of these offers have to match though.
    There are 10 products resulting from google shopping that were labeled by the turks even after removing items with
    different counts or other obvious non-matches. Let's say google shopping returns 67 results, then first obvious
    non-matches are removed and top 10 similar results are selected after that. (no threshold is applied)

    Maybe the thing should also take into account that Fyndiq is a marketplace. So multiple Fyndiq offers may map to the
    same physical product. So it may not be a good idea to take multiple products from Fyndiq in the same batch

    As I was writing the thing above I realized it is impossible not to have multiple Fyndiq products in the same batch
    because then there is no more data to put in the batch at all. But this case should be handled by the fact that
    Fyndiq products are used as targets, an offer from the same product on Fyndiq would generate similar candidates
    on Google Shopping and then there is a link through those offers from Google Shopping that are commonly matching
    both appearances on Fyindiq.
    """

    @staticmethod
    def __sample_pair(df: DataFrame) -> Tuple[pd.Series, pd.Series, int]:
        anchor_cluster_selection = df.sample(2)
        first_matching_offer, second_matching_offer = \
            anchor_cluster_selection.iloc[0].copy(), anchor_cluster_selection.iloc[1].copy()

        return first_matching_offer, second_matching_offer, first_matching_offer['cluster_id']

    def __call__(self, x):
        features_left = []
        features_right = []
        labels = []

        # TODO: use pandas logic instead of looping through (maybe)
        for df in x:
            anchor_offer = df[df['source'] == '#1'].iloc[0].copy()
            anchor_cluster = df[df['cluster_id'] == anchor_offer['cluster_id']]
            if len(anchor_cluster) < 2:
                features_left.append(anchor_offer['text'])
                features_right.append(anchor_offer['text'])
                labels.append(anchor_offer['cluster_id'])
            else:
                sample_left, sample_right, label = self.__sample_pair(anchor_cluster)

                features_left.append(sample_left)
                features_right.append(sample_right)
                labels.append(label)

            if len(anchor_cluster) >= 3:
                sample_left, sample_right, label = self.__sample_pair(anchor_cluster)

                features_left.append(sample_left)
                features_right.append(sample_right)
                labels.append(label)

            if len(anchor_cluster) >= 5:
                sample_left, sample_right, label = self.__sample_pair(anchor_cluster)

                features_left.append(sample_left)
                features_right.append(sample_right)
                labels.append(label)

            non_matching_pool = df[df['cluster_id'] != anchor_offer['cluster_id']]
            if len(non_matching_pool) > 0:
                non_matching_offer = non_matching_pool.sample(1).iloc[0].copy()

                features_left.append(non_matching_offer['text'])
                features_right.append(non_matching_offer['text'])
                labels.append(non_matching_offer['cluster_id'])

        batch_left = self.tokenize_features(features_left)
        batch_right = self.tokenize_features(features_right)

        return self.collate_pair(batch_left, batch_right, labels)


class TurksDataContrastivePredictor(ContrastivePredictor):
    def instantiate_pretrain_collator(self, arguments: ExperimentsArgumentParser):
        return ContrastivePretrainingTurksDatasetCollator(tokenizer=self.tokenizer,
                                                          max_length=self.config.max_tokens)

    @staticmethod
    def instantiate_pretrain_dataset(df: DataFrame, source_aware_sampling: bool = True):
        return TurkLabeledDataset(df)


