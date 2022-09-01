from sklearn.model_selection import train_test_split

from evaluation import ModelEvaluator
from models import (
    PopularityRecommendationModel,
    PopularityYearRecommendationModel,
    ContentBasedRecommendationModel,
    CollaborativeRecommendationModel, HybridRecommendationModel
)
from utils.dataset_utils import preprocess_books, preprocess_ratings, \
    drop_irrelevant_books
from utils.df_utils import (
    load_books_dataset,
    load_ratings_dataset,
    get_users_subset, intersect_df,
)

books, dropped_books = preprocess_books(
    load_books_dataset("dataset/Books.csv"),
    drop_irrelevant_publishers=100,
    drop_irrelevant_authors=50
)
ratings = load_ratings_dataset("dataset/Ratings.csv", drop_books=dropped_books)

print("Dataset Loaded...")

full_train_data, full_test_data = train_test_split(ratings, test_size=0.15)

full_train_data = preprocess_ratings(full_train_data)
full_test_data = preprocess_ratings(full_test_data)

content_new_data = get_users_subset(full_train_data, full_test_data, 500)

print("Dataset Splitted...")

evaluator = ModelEvaluator(
    training_data=content_new_data["train_data"],
    testing_data=content_new_data["test_data"],
    favourite_threshold=1,
)

print("Evaluator Loaded...")


# popularity_rec_model = PopularityRecommendationModel(
#     books, test_data, threshold=1
# )
# print(f"{popularity_rec_model.MODEL_NAME} Loaded...")
#
# popularity_year_rec_model = PopularityYearRecommendationModel(
#     books, train_data, threshold=1
# )
# print(f"{popularity_year_rec_model.MODEL_NAME} Loaded...")
#
# pop_global_metrics, _ = evaluator.evaluate_model(
#     popularity_rec_model,
#     multiprocessing=False
# )
# print(pop_global_metrics)
#
# pop_year_global_metrics, _ = evaluator.evaluate_model(
#     popularity_year_rec_model,
#     multiprocessing=False
# )
# print(pop_year_global_metrics)


# content_rec_model = ContentBasedRecommendationModel(
#     column_name="publisher", books_df=books, ratings_df=new_data["train_data"]
# )
# print(f"{content_rec_model.MODEL_NAME} Loaded...")
#
# content_global_metrics, _ = evaluator.evaluate_model(
#     content_rec_model,
#     multiprocessing=False
# )

# ratings_df = preprocess_ratings(ratings, drop_books=dropped_books)

collab_ratings_df = drop_irrelevant_books(ratings, 200)

collab_full_train_data, collab_full_test_data = train_test_split(
    collab_ratings_df, test_size=0.15
)

collab_full_train_data = preprocess_ratings(collab_full_train_data)
collab_full_test_data = preprocess_ratings(collab_full_test_data)

collab_new_data = get_users_subset(
    collab_full_train_data, collab_full_test_data, 500
)

collaborative_evaluator = ModelEvaluator(
    training_data=collab_new_data["train_data"],
    testing_data=collab_new_data["test_data"],
    favourite_threshold=1,
)

# collaborative_rec_model = CollaborativeRecommendationModel(
#     ratings_df=ratings_df
# )

# collaborative_global_metrics, _ = collaborative_evaluator.evaluate_model(
#     collaborative_rec_model
# )

hybrid_new_train_data, _ = intersect_df(
    content_new_data["train_data"], collab_new_data["train_data"], "book_id"
)

hybrid_new_test_data, _ = intersect_df(
    content_new_data["test_data"], collab_new_data["test_data"], "book_id"
)

hybrid_evaluator = ModelEvaluator(
    training_data=hybrid_new_train_data,
    testing_data=hybrid_new_test_data,
    favourite_threshold=1,
)

hybrid_rec_model = HybridRecommendationModel(
    content_column_name="author",
    content_books_df=books,
    content_ratings_df=content_new_data["train_data"],
    collab_ratings_df=collab_ratings_df
)

hybrid_global_metrics, _ = hybrid_evaluator.evaluate_model(
    hybrid_rec_model
)

