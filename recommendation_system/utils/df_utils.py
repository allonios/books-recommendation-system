import random
from typing import Tuple, List, Dict, Set

import numpy as np
import pandas as pd


def handle_messing_year_values(path: str) -> None:
    """
    A function to load the books dataset and remove rows that has missing
    year values without commas, example for that case:
    id1, id2, id3, id4
    1, 2, 3, 4
    1, 2, 3, 4
    1, 2, 4
    1, 2,, 4
    the third row has only three values and can't be detected like row 4
    missing value.
    :param path:
    :return:
    """
    df = pd.read_csv(path)
    isbn_to_remove = set(
        map(
            lambda row: row[0],
            filter(
                lambda row: isinstance(row[3], str) and (
                    not row[3].isnumeric()),
                df.iloc
            )
        )
    )
    print("The following rows will be removed:")
    print(isbn_to_remove)

    df = df[~df["ISBN"].isin(isbn_to_remove)]
    df.to_csv(path, index=False)


def load_books_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=(
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Year-Of-Publication",
            "Publisher",
        ),
        dtype={"Year-Of-Publication": np.uint16}
    ).rename(
        columns={
            "ISBN": "id",
            "Book-Title": "title",
            "Book-Author": "author",
            "Year-Of-Publication": "year",
            "Publisher": "publisher",
        }
    )

    return df


def load_ratings_dataset(path: str, drop_books: Set = None) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={
            "User-ID": "user_id",
            "ISBN": "book_id",
            "Book-Rating": "rating",
        }
    )

    if drop_books:
        df = df[~df["book_id"].isin(drop_books)]

    return df


def get_average_age_for_country(
        users_df: pd.DataFrame, country_name: str,
) -> int:
    user_country_rows = users_df[users_df["location"] == country_name]

    if len(user_country_rows) == 0:
        return 0
    return round(sum(user_country_rows["age"]) / len(user_country_rows))


def load_users_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={
            "User-ID": "id",
            "Location": "location",
            "Age": "age",
        }
    )

    def get_country_name(location: str) -> str:
        detailed_location = location.split(",")
        return detailed_location[len(detailed_location) - 1]

    # make location simpler, use only counter name.
    df["location"] = df["location"].apply(
        lambda location: get_country_name(location)
    )

    # splitting DF to a list of country based DFs.
    country_set = set(df["location"])
    df_list = []
    for country in country_set:
        df_list.append(df[df["location"] == country])

    # file each country based DF age nan values with mean.
    for country_df in df_list:
        country_df.fillna(round(country_df["age"].mean()), inplace=True)

    # merge dfs.
    df = pd.concat(df_list)

    return df


def slice_df_by_column_name(
        df: pd.DataFrame,
        column_name: str,
        start_value: int = None,
        end_value: int = None,
) -> pd.DataFrame:
    """
    A function to get a slice of a dataframe by specifying value ranges.
    :param df:
    :param column_name:
    :param start_value:
    :param end_value:
    :return:
    """
    if not start_value:
        return df[df[column_name] < end_value]
    if not end_value:
        return df[start_value <= df[column_name]]
    return df[df[column_name].between(start_value, end_value)]


def train_test_split(
        books_df: pd.DataFrame,
        ratings_df: pd.DataFrame,
        start_year: int = None,
        end_year: int = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A function similar to scikit-lear train_test_split, but it is based on a
    book publication year range.
    :param books_df:
    :param ratings_df:
    :param start_year:
    :param end_year:
    :return:
    """
    books_ids = slice_df_by_column_name(
        df=books_df,
        column_name="year",
        start_value=start_year,
        end_value=end_year
    )["id"].values

    training_data = ratings_df[ratings_df["book_id"].isin(books_ids)]
    testing_data = ratings_df[~ratings_df["book_id"].isin(books_ids)]

    print("Split Percentage:")
    print("Books:")
    print("-----------------------------------------------------------------")
    print(
        f"Train Percentage: {(len(books_ids) * 100) / len(books_df)}"
    )
    print(
        "Test Percentage: "
        f"{((len(books_df) - len(books_ids)) * 100) / len(books_df)}"
    )
    print("-----------------------------------------------------------------")
    print("Ratings:")
    print("-----------------------------------------------------------------")
    print(
        f"Train Percentage: {(len(training_data) * 100) / len(ratings_df)}"
    )
    print(
        "Test Percentage: "
        f"{(len(testing_data) * 100) / len(ratings_df)}"
    )
    print("-----------------------------------------------------------------")

    return training_data, testing_data

def get_users_subset(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        test_samples_count: int,
) -> Dict:
    train_users = set(train_data["user_id"])
    test_users = set(test_data["user_id"])
    common_users = train_users.intersection(test_users)
    sub_users_set = set(
        random.sample(
            list(common_users), test_samples_count
        )
    )

    return {
        "train_data": train_data[train_data["user_id"].isin(sub_users_set)],
        "test_data": test_data[test_data["user_id"].isin(sub_users_set)],
        "common_users": sub_users_set,
    }


def intersect_df(
        df1: pd.DataFrame,
        df2: pd.DataFrame, column_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    df1_set = set(df1[column_name])
    df2_set = set(df2[column_name])

    common_data = df1_set.intersection(df2_set)

    new_df1 = df1[df1[column_name].isin(common_data)]
    new_df2 = df2[df2[column_name].isin(common_data)]

    return new_df1, new_df2





