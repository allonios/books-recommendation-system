from collections import Counter
from typing import Set, Union, Tuple

import numpy as np
import pandas as pd
from numba import jit
from sklearn.preprocessing import MinMaxScaler

from utils.df_utils import slice_df_by_column_name


def filter_elements(
        df: pd.DataFrame, column_name: str, exclude_items: Set
) -> pd.DataFrame:
    return df[~df[column_name].isin(exclude_items)]


def weighted_rating(row, m, c):
    """
    A function to compute the popularity for books using the following equation
    (v/(v+m) * r) + (m/(m+v) * c)
    where:
    v is the number of ratings for the movie.
    m is the minimum rating required to be listed in the chart.
    r is the average rating of the movie.
    c is the mean ratings across the whole movies.
    :return:
    """
    # rating count.
    v = row["rating_count"]
    # rating mean.
    r = row["rating_mean"]
    # Calculation based on the IMDB formula.
    try:
        return (v / (v + m) * r) + (m / (m + v) * c)
    except TypeError:
        breakpoint()


@jit(nopython=True)
def compute_popularity_from_array(
        rating_count_mean: np.ndarray,
        m: float,
        c: float
):
    """
    A function to manually compute popularity for books
    using the following equation:
    (v/(v+m) * r) + (m/(m+v) * c)
    where:
    v is the number of ratings for the book.
    m is the minimum rating required to be listed in the chart.
    r is the average rating of the book.
    c is the mean ratings across the whole books.
    :return: popularity array.
    """
    popularity = np.zeros(rating_count_mean.shape[0])

    index = 0
    for element in rating_count_mean:
        v = element[0]
        r = element[1]
        popularity[index] = (v / (v + m) * r) + (m / (m + v) * c)
        index += 1

    return popularity


def compute_popularity(
        books_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        m: float,
) -> pd.DataFrame:
    """
    :param books_df:
    :param ratings_df:
    :param m: the minimum rating required to be listed in the chart.
    :return: a dataframe with the popularity column.
    """
    total_df = pd.merge(
        books_df, ratings_df, left_on="id", right_on="book_id"
    )

    total_df = total_df.drop(
        columns=["title", "author", "publisher", "user_id", "book_id", ]
    )

    total_df = total_df.groupby("id").aggregate(
        {
            "rating": ["mean", "count"],
            "year": ["first"],
        }
    ).reset_index()

    total_df.columns = ["id", "rating_mean", "rating_count", "year", ]

    total_df = total_df.sort_values(by="rating_mean", ascending=False)

    c = total_df["rating_mean"].mean()

    rating_data = total_df[["rating_count", "rating_mean"]].to_numpy(
        dtype=np.float32
    )

    total_df["popularity"] = compute_popularity_from_array(rating_data, m, c)

    return total_df.sort_values("popularity", ascending=False)


def compute_year_popularity(
        popularity_df: pd.DataFrame,
        oldest_book_year: int
) -> pd.DataFrame:
    """
    A function to compute popularity based on year, this function assumes that
    the popularity has already been computed.
    it follows this equation:
    popularity * (book_year - oldest_book_year)
    :return: a dataframe with the popularity_year column.
    """
    popularity_df["popularity_year"] = (
            popularity_df["popularity"]
            * (popularity_df["year"] - oldest_book_year)
    )

    # normalize values to base value 10.
    scaler = MinMaxScaler()
    popularity_df["popularity_year"] = scaler.fit_transform(
        popularity_df["popularity_year"].values.reshape(-1, 1)
    )
    popularity_df["popularity_year"] = popularity_df["popularity_year"] * 10

    return popularity_df.sort_values("popularity_year", ascending=False)


def get_user_favorite_books(
        user_id: int,
        ratings_df: pd.DataFrame,
        threshold: float
) -> Set[str]:
    return set(
        ratings_df[
            (ratings_df["user_id"] == user_id)
            & (ratings_df["rating"] >= threshold)
            ].sort_values(
            "rating", ascending=False
        )["book_id"].values
    )


def preprocess_books(
        books_df: pd.DataFrame,
        drop_irrelevant_publishers: Union[int, bool] = False,
        drop_irrelevant_authors: Union[int, bool] = False,
) -> Tuple[pd.DataFrame, Set]:
    """
    Books dataset preprocessor.
    :param books_df:
    :param drop_irrelevant_publishers: a threshold for publishers occurrences,
    if a publisher occurrences is below the threshold,
    all of its books will be dropped.
    :param drop_irrelevant_authors: a threshold for authors occurrences,
    if an author occurrences is below the threshold,
    all of its books will be dropped.
    :return:
    """
    # drop nan values.
    new_books_df = books_df.dropna(subset=["author", "publisher"])

    # removing books that has year = 0.
    new_books_df = slice_df_by_column_name(new_books_df, "year", 1)

    # removing books that has an Unknown Publisher.
    new_books_df = new_books_df[
        (
            new_books_df["publisher"]
            != "Unknown Publisher - Being Researched"
        )
        & (new_books_df["publisher"] != "Unknown")
    ]

    # removing books that has an Unknown Author.
    new_books_df = new_books_df[
        ~new_books_df["author"].str.contains("unknown", regex=False)
    ]

    if drop_irrelevant_publishers:
        publishers_count = dict(Counter(new_books_df["publisher"]))
        publishers = set(
            filter(
                lambda pub: publishers_count[pub] > drop_irrelevant_publishers,
                publishers_count
            )
        )
        new_books_df = new_books_df[new_books_df["publisher"].isin(publishers)]

    if drop_irrelevant_authors:
        authors_count = dict(Counter(new_books_df["author"]))
        authors = set(
            filter(
                lambda pub: authors_count[pub] > drop_irrelevant_authors,
                authors_count
            )
        )
        new_books_df = new_books_df[new_books_df["author"].isin(authors)]

    new_books_df.set_index(keys=["id"])

    full_books_ids = set(books_df["id"])
    new_books_ids = set(new_books_df["id"])

    return new_books_df, full_books_ids.difference(new_books_ids)


def preprocess_ratings(
        ratings_df: pd.DataFrame, drop_books: Set = None
) -> pd.DataFrame:
    # drop nan values.
    new_ratings_df = ratings_df.dropna()

    # drop books that was dropped in the books preprocessing.
    if drop_books:
        new_ratings_df = new_ratings_df[
            ~new_ratings_df["book_id"].isin(drop_books)
        ]


    # drop users that has a total ratings sum of 0.
    users_ratings_sums = ratings_df.groupby("user_id").aggregate(
        {
            "rating": ["sum"]
        }
    ).reset_index()
    users_ratings_sums.columns = ["user_id", "ratings_sum"]
    zero_ratings_users = users_ratings_sums[
        users_ratings_sums["ratings_sum"] == 0
        ]["user_id"]

    new_ratings_df.loc[
        new_ratings_df["user_id"].isin(zero_ratings_users), "rating"
    ] = 0.0001

    return new_ratings_df

def build_column_vectorizer(
        df: pd.DataFrame, column_name: str
) -> pd.DataFrame:
    """
    A function to add something like a one-hot-encoded values based on a column
    distinct values
    example (column name is "A"):
    A   B   C   a   b   c
    a   -   -   1   0   0
    b   -   -   0   1   0
    c   -   -   0   0   1
    a   -   -   1   0   0
    a   -   -   1   0   0
    a   -   -   1   0   0
    b   -   -   0   1   0
    :param df:
    :param column_name:
    :return:
    """
    import warnings
    from pandas.errors import PerformanceWarning
    warnings.filterwarnings("ignore", category=PerformanceWarning)

    distinct_column_values = set(df[column_name])

    expended_df = df.copy()
    for value in distinct_column_values:
        new_values = np.isin(
            expended_df[column_name].values, value
        ).astype(np.uint8)
        expended_df[value] = new_values

    return expended_df


def drop_irrelevant_books(
        ratings_df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    """
    A function to remove books that has a total rating below a
    certain threshold
    :param ratings_df:
    :param threshold:
    :return:
    """
    books_ratings_sums = ratings_df.groupby("book_id").aggregate(
        {
            "rating": ["sum"]
        }
    ).reset_index()

    books_ratings_sums.columns = ["book_id", "rating"]

    books_ids_to_drop = books_ratings_sums[
        books_ratings_sums["rating"] > threshold
    ]["book_id"]

    # sorted_books_ratings_sums = books_ratings_sums.sort_values(
    #     by=["rating"], ascending=False
    # )
    return ratings_df[ratings_df["book_id"].isin(books_ids_to_drop)]
