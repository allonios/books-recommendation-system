from dataclasses import dataclass
from functools import partial
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from models import BaseRecommendationModel
from utils.dataset_utils import get_user_favorite_books


@dataclass
class ModelEvaluator:
    training_data: pd.DataFrame
    testing_data: pd.DataFrame
    favourite_threshold: float

    def evaluate_model_for_user(
            self,
            user_id: int,
            model: BaseRecommendationModel,
    ) -> Dict:
        favorites_in_test = get_user_favorite_books(
            user_id, self.testing_data, self.favourite_threshold
        )
        favorites_in_train = get_user_favorite_books(
            user_id, self.training_data, self.favourite_threshold
        )

        recomms_df = model.recommend_items(
            user_id=user_id, items_to_ignore=favorites_in_train, topn=10
        )

        true_relevant_count = recomms_df[
            recomms_df["id"].isin(favorites_in_test)].shape[0]

        top_five_recommended = recomms_df.head(5)
        top_ten_recommended = recomms_df.head(10)

        hits_at_five_count = top_five_recommended[
            top_five_recommended["id"].isin(favorites_in_test)
        ].shape[0]

        hits_at_ten_count = top_ten_recommended[
            top_ten_recommended["id"].isin(favorites_in_test)
        ].shape[0]

        precision_at_five = (
            hits_at_five_count / top_five_recommended.shape[0]
            if top_five_recommended.shape[0] != 0
            else 1
        )
        recall_at_five = (
            hits_at_five_count / true_relevant_count
            if true_relevant_count != 0
            else 1
        )

        precision_at_ten = (
            hits_at_ten_count / top_ten_recommended.shape[0]
            if top_ten_recommended.shape[0] != 0
            else 1
        )
        recall_at_ten = (
            hits_at_ten_count / true_relevant_count
            if true_relevant_count != 0
            else 1
        )

        person_metrics = {
            "hits@5_count": hits_at_five_count,
            "hits@10_count": hits_at_ten_count,
            "recommended@5_count": top_five_recommended.shape[0],
            "recommended@10_count": top_ten_recommended.shape[0],
            "relevant_count": true_relevant_count,
            "recall@5": recall_at_five,
            "recall@10": recall_at_ten,
            "precision@5": precision_at_five,
            "precision@10": precision_at_ten
        }
        return person_metrics

    def evaluate_model(
            self,
            model: BaseRecommendationModel,
            multiprocessing: bool = False
    ) -> Tuple[Dict, Dict]:
        result = []

        print(f"Running Evaluation for {model.MODEL_NAME}")
        if multiprocessing:
            result = process_map(
                partial(self.evaluate_model_for_user, model=model),
                set(self.testing_data["user_id"]),
                chunksize=10,
                max_workers=8
            )
            # with Pool(8) as pool:
            #     result = pool.map(
            #         partial(self.evaluate_model_for_user, model=model),
            #         self.test_users_ids,
            #     )
        else:
            with tqdm(total=len(set(self.testing_data["user_id"]))) as pbar:
                for user_id in set(self.testing_data["user_id"]):
                    result.append(self.evaluate_model_for_user(user_id, model))
                    pbar.update(1)

        print(f"processed {len(set(self.testing_data['user_id']))} users")
        print(f"Finished {model.MODEL_NAME} Evaluation...")

        detailed_results_df = pd.DataFrame(result).sort_values(
            "hits@5_count", ascending=False
        )

        # relevant_count = float(detailed_results_df["relevant_count"].sum())
        # if not relevant_count:
        #     global_recall_at_5 = global_recall_at_10 = 1
        # else:
        global_recall_at_5 = (
                detailed_results_df["hits@5_count"].sum()
                / float(detailed_results_df["relevant_count"].sum())
        )
        global_recall_at_10 = (
                detailed_results_df["hits@10_count"].sum()
                / float(detailed_results_df["relevant_count"].sum())
        )

        global_precision_at_5 = (
                detailed_results_df["hits@5_count"].sum()
                / float(detailed_results_df["recommended@5_count"].sum())
        )
        global_precision_at_10 = (
                detailed_results_df["hits@10_count"].sum()
                / float(detailed_results_df["recommended@10_count"].sum())
        )

        global_metrics = {
            "model_name": model.MODEL_NAME,
            "recall@5": global_recall_at_5,
            "recall@10": global_recall_at_10,
            "precision@5": global_precision_at_5,
            "precision@10": global_precision_at_10
        }
        return global_metrics, detailed_results_df
