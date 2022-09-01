from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Set, List

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity

from utils.dataset_utils import (
    compute_popularity,
    compute_year_popularity,
    build_column_vectorizer
)
from utils.utils import normalize


class BaseRecommendationModel(metaclass=ABCMeta):
    MODEL_NAME = "Base"

    @abstractmethod
    def recommend_items(
            self,
            user_id: int,
            items_to_ignore: Set = (),
            topn: int = 10,
    ) -> pd.DataFrame:
        ...


@dataclass
class PopularityRecommendationModel(BaseRecommendationModel):
    MODEL_NAME = "Popularity Recommendation Model"

    def __init__(
            self,
            books_df: pd.DataFrame,
            ratings_df: pd.DataFrame,
            threshold: float
    ) -> None:
        self.books_df = books_df
        self.ratings_df = ratings_df
        self.popularity_df = compute_popularity(
            books_df, ratings_df, threshold
        )

    def recommend_items(
            self,
            user_id: int = None,
            items_to_ignore: Set = (),
            topn: int = 10,
    ) -> pd.DataFrame:
        return self.popularity_df[
            ~self.popularity_df["id"].isin(items_to_ignore)
        ].head(topn)


class PopularityYearRecommendationModel(PopularityRecommendationModel):
    MODEL_NAME = "Popularity Year Based Recommendation Model"

    def __init__(
            self,
            books_df: pd.DataFrame,
            ratings_df: pd.DataFrame,
            threshold: float
    ) -> None:
        super().__init__(books_df, ratings_df, threshold)
        self.popularity_df = compute_year_popularity(
            popularity_df=self.popularity_df,
            oldest_book_year=min(self.books_df["year"])
        )


class ContentBasedRecommendationModel(BaseRecommendationModel):
    MODEL_NAME = "Content Based Model"

    def __init__(
            self,
            column_name: str,
            books_df: pd.DataFrame,
            ratings_df: pd.DataFrame
    ) -> None:
        self.column_name = column_name
        self.books_df = build_column_vectorizer(books_df, column_name)
        self.features_names = set(books_df[column_name])
        self.books_features = self.books_df[self.features_names]
        self.ratings_df = ratings_df

        import warnings
        warnings.simplefilter("ignore", category=FutureWarning)

    def get_book_profile(self, book_id: str) -> pd.DataFrame:
        return self.books_df[
            self.books_df["id"] == book_id
            ][self.features_names]

    def get_books(self, books_ids: Set[str]) -> pd.DataFrame:
        return self.books_df[self.books_df["id"].isin(books_ids)]

    def build_user_profile(self, user_id: int) -> np.ndarray:
        user_df = self.ratings_df[self.ratings_df["user_id"] == user_id]
        user_books = self.get_books(user_df["book_id"].values)

        user_books_profiles = user_books[self.features_names].values
        user_books_ratings = user_df["rating"].values.reshape(-1, 1)

        if user_books_profiles.shape[0] < user_books_ratings.shape[0]:
            user_books_profiles = np.concatenate(
                [
                    user_books_profiles,
                    np.full(
                        (
                            (
                                user_books_ratings.shape[0]
                                - user_books_profiles.shape[0]
                            ),
                            user_books_profiles.shape[1]
                        ),
                        0.0001
                    )
                ],
                axis=0
            )

        if not user_books_ratings.size:
            print("raised from empty")
            breakpoint()
            raise ValueError()

        if not any(user_books_ratings):
            print("raised from zeros")
            breakpoint()
            raise ValueError()

        try:
            user_profile = np.sum(
                np.multiply(
                    user_books_profiles,
                    user_books_ratings,
                ),
                axis=0
            ) / np.sum(user_books_ratings)
        except:
            print("from divide")
            breakpoint()


        return user_profile

    def get_similar_books_to_user_profile(
            self, user_id: int, topn: int = 10
    ) -> List:
        user_profile = self.build_user_profile(user_id).reshape(1, -1)
        # Computes the cosine similarity between the user profile and
        # all books profiles.
        cosine_similarities = cosine_similarity(
            user_profile, self.books_features
        )
        # Gets the top similar items.
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_items = sorted(
            [
                (
                    self.books_df.iloc[i]["id"],
                    cosine_similarities[0, i]
                ) for i in similar_indices
            ],
            key=lambda x: -x[1]
        )
        return similar_items

    def recommend_items(
            self, user_id: int,
            items_to_ignore: Set = (),
            topn: int = 10
    ) -> pd.DataFrame:
        similar_items = self.get_similar_books_to_user_profile(user_id)
        similar_items_filtered = list(
            filter(
                lambda x: x[0] not in items_to_ignore,
                similar_items
            )
        )

        if topn == -1:
            recommendations_df = pd.DataFrame(
                similar_items_filtered,
                columns=["id", "predicted_rating"]
            )
        else:
            recommendations_df = pd.DataFrame(
                similar_items_filtered,
                columns=["id", "predicted_rating"]
            ).head(topn)

        recommendations_df["predicted_rating"] = recommendations_df[
            "predicted_rating"
        ].apply(lambda x: normalize(x, 1.0, 0.0))

        return recommendations_df


class CollaborativeRecommendationModel(BaseRecommendationModel):
    MODEL_NAME = "Collaborative Filtering"

    def __init__(
            self, ratings_df: pd.DataFrame, NUMBER_OF_FACTORS_MF: int = 15
    ) -> None:
        users_items_pivot_matrix_df = ratings_df.pivot(
            index="user_id",
            columns="book_id",
            values="rating"
        ).fillna(0)


        # Performs matrix factorization of the original user item matrix
        # U, sigma, Vt = svds(users_items_pivot_matrix, k = NUMBER_OF_FACTORS_MF)
        U, sigma, Vt = svds(
            users_items_pivot_matrix_df.values,
            k=NUMBER_OF_FACTORS_MF
        )
        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

        # Converting the reconstructed matrix back to a Pandas dataframe
        self.predictions_df = pd.DataFrame(
            all_user_predicted_ratings,
            columns=users_items_pivot_matrix_df.columns,
            index=users_items_pivot_matrix_df.index
        )

        self.predictions_df = self.predictions_df.apply(
            lambda x: normalize(
                x,
                all_user_predicted_ratings.max(),
                all_user_predicted_ratings.min()
            )
        )

    def recommend_items(
            self, user_id: int,
            items_to_ignore: Set = (),
            topn: int = 10
    ) -> pd.DataFrame:
        # Get and sort the user's predictions
        sorted_user_predictions = self.predictions_df.loc[user_id].sort_values(
            ascending=False
        )
        recommendations = {
            "book_id": sorted_user_predictions.index,
            "predicted_rating": sorted_user_predictions.values
        }
        recommendations_df = pd.DataFrame(recommendations)

        # Recommend the highest predicted rating books that the user hasn't
        # seen yet.
        if topn == -1:
            recommendations_df = recommendations_df[
                ~recommendations_df["book_id"].isin(items_to_ignore)
            ].sort_values('predicted_rating', ascending=False)
        else:
            recommendations_df = recommendations_df[
                ~recommendations_df["book_id"].isin(items_to_ignore)
            ].sort_values('predicted_rating', ascending=False).head(topn)

        return recommendations_df.rename(
            columns={"book_id": "id"}
        )


class HybridRecommendationModel(BaseRecommendationModel):
    MODEL_NAME = "Hybrid Recommendation Model"

    def __init__(
            self,
            content_column_name: str,
            content_books_df: pd.DataFrame,
            content_ratings_df: pd.DataFrame,
            collab_ratings_df: pd.DataFrame,
            NUMBER_OF_FACTORS_MF: int = 15
    ) -> None:
        self.model1 = ContentBasedRecommendationModel(
            content_column_name, content_books_df, content_ratings_df
        )
        self.model2 = CollaborativeRecommendationModel(
            collab_ratings_df, NUMBER_OF_FACTORS_MF
        )

    def recommend_items(
            self,
            user_id: int,
            items_to_ignore: Set = (),
            topn: int = 10
    ) -> pd.DataFrame:
        model1_recommendations = self.model1.recommend_items(
            user_id, items_to_ignore, topn=-1
        )
        model2_recommendations = self.model2.recommend_items(
            user_id, items_to_ignore, topn=-1
        )

        model1_recommendations_books_ids = set(model1_recommendations["id"])
        model2_recommendations_books_ids = set(model2_recommendations["id"])

        common_recommendations = model1_recommendations_books_ids.intersection(
            model2_recommendations_books_ids
        )

        model1_recommendations = model1_recommendations[
            ~model1_recommendations["id"].isin(common_recommendations)
        ]
        model2_recommendations = model2_recommendations[
            ~model2_recommendations["id"].isin(common_recommendations)
        ]

        model1_recommendations = model1_recommendations.sort_values(
            by=["predicted_rating"], ascending=False
        )

        model2_recommendations = model2_recommendations.sort_values(
            by=["predicted_rating"], ascending=False
        )

        return pd.concat(
            [
                model1_recommendations.head(int(topn / 2)),
                model2_recommendations.head(int(topn / 2))
            ]
        )
