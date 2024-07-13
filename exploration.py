# %%
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

import numpy as np

# %%
books_df = pd.read_csv("./data/zygmuntz_books.csv", on_bad_lines="skip")

# %%
# Get books with >= 10,000 ratings
books_df = books_df[books_df["ratings_count"] >= 10000]


# %%
# Scale the rating of all the books

scaler = StandardScaler()
books_df["scaled_rating"] = scaler.fit_transform(X=books_df[["average_rating"]])

# %%
# Get books with series information

# Books in series have parenthetical information that includes the number of the series
books_with_numbers_df = books_df[books_df["title"].str.contains("\#")].copy()

# Remove any books that are part of multiple series; they probably aren't trilogies.
books_with_numbers_df = books_with_numbers_df[
    ~books_with_numbers_df["title"].str.contains(r"\(.*\;.*\)")
]

# Add series and series position columns
books_with_numbers_df["series"] = books_with_numbers_df["title"].str.extract(
    r"\((.*?),"
)
books_with_numbers_df["series_position"] = books_with_numbers_df["title"].str.extract(
    r"\#(.*?)\)"
)

# Remove Half books, they're cash-ins and we don't care. This will also remove some comics which I'm not intersted in either
books_with_numbers_df = books_with_numbers_df[
    books_with_numbers_df["series_position"].str.match(r"^[1-9]$").astype(bool)
]

# %%
# Filter down to trilogies

series_length = books_with_numbers_df.groupby("series").size()
trilogies_list = series_length[series_length == 3]

trilogies_df = books_with_numbers_df[
    np.isin(books_with_numbers_df["series"], trilogies_list.index)
]


# %% Some manual cleaning

# All series with entries greater than 3 aren't actually trilogies.
not_trilogies = trilogies_df[
    ~np.isin(trilogies_df["series_position"], ["1", "2", "3"])
]["series"].unique()
trilogies_df = trilogies_df[~np.isin(trilogies_df["series"], not_trilogies)]

# The above issue seemed to be caused by some series having different series in different titles.
# E.g., there was Hitchhiker's Guide and Hitchhikers Guide to the Galaxy

# These are two different series with the same name.
trilogies_df = trilogies_df[trilogies_df["series"] != "Between the Lines"]

trilogy_scores = trilogies_df[
    ["series", "series_position", "average_rating", "scaled_rating"]
].sort_values(["series", "series_position"])
# %%
# Start plotting

initial_figure = px.line(
    trilogy_scores,
    x="series_position",
    y="average_rating",
    color="series",
    labels={"average_rating": "GoodReads Rating", "series_position": "Book Number"},
    title="Trilogy Book Rating over Book Number",
)

initial_figure.update_layout(showlegend=False)
initial_figure.show()

initial_figure.write_image(
    "./results/images/initial_figure.png", height=720, width=1080
)

# %%
# Give our trilogies the same origin
trilogy_first_book_scores = trilogy_scores.groupby("series")["scaled_rating"].first()
trilogy_first_book_dict = trilogy_first_book_scores.to_dict()


series_first_book_lookup_lambda = lambda series_name: trilogy_first_book_dict[
    series_name
]
trilogy_scores["first_book_score"] = trilogy_scores["series"].apply(
    series_first_book_lookup_lambda
)

trilogy_scores["centered_score"] = (
    trilogy_scores["scaled_rating"] - trilogy_scores["first_book_score"]
)

# %%
# Plot centered
centered_figure = px.line(
    trilogy_scores,
    x="series_position",
    y="centered_score",
    color="series",
    labels={"centered_score": "Centered Rating", "series_position": "Book Number"},
    title="Trilogy Book Rating over Book Number",
)

# centered_figure.update_layout(showlegend=False)
centered_figure.show()
# %%

overall_average_rating = books_df["average_rating"].mean()
print(
    f"The average rating for all books is {overall_average_rating:.2f}, which is a little high."
)


StandardScaler().fit_transform(books_df["average_rating"])
# %%

# %%
# read IMDb data
# name_basics_df = pd.read_csv("./data/name.basics.tsv", sep='\t')
title_basics_df = pd.read_csv("./data/title.basics.tsv", sep="\t")

# %%
# Filter to movie
movie_df = title_basics_df[
    (title_basics_df["titleType"] == "movie") & (title_basics_df["isAdult"] == 0)
]
