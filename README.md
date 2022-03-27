# Prediction of movie profitability

Authors : Kévin Assobo Baguéka, Antoine Gey, Robin Labbé, Ewan Sean, Joshua Wolff

The aim of this project is to predict the rentability of a movie using its characteristics like budget, genre, cast and realization crew. We make the assumption that the producer want a minimum profitability ratio (lets say 2 : the film's revenue must be at least two times bigger than its budget). Films can now be separated in two classes : those that are enough lucrative and those that are not. The problem is then reduced at a binary classification problem.

The data comes from `The Movie DataBase API (TMDB)`, but is not endorsed or certified by TMDB. See https://developers.themoviedb.org/3/getting-started/introduction for more information.

## Getting started

### Install

To run a submission and the notebook you will need the dependencies listed
in `requirements.txt`. You can install install the dependencies with the
following command-line:

```bash
pip install -U -r requirements.txt
```

### Challenge description

Get started with the [dedicated notebook](movie_profitability_prediction_starting_kit.ipynb)


### Test a submission

The submissions need to be located in the `submissions` folder. For instance
for `my_submission`, it should be located in `submissions/my_submission`.

To run a specific submission, you can use the `ramp-test` command line:

```bash
ramp-test --submission my_submission
```

You can get more information regarding this command line:

```bash
ramp-test --help
```

### To go further

You can find more information regarding `ramp-workflow` in the
[dedicated documentation](https://paris-saclay-cds.github.io/ramp-docs/ramp-workflow/stable/using_kits.html)
