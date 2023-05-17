# ♥Netflix recommendation System Using SVD♥

○ Recommendation Engine-A filtering system that seeks to predict and show the items of user interest.It is a data filtering tool that make use of algorithms and data to recommend the most relevant items to a particular user.It is mostly used in digital domains

♦ DATASET ♦

○ Link -https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data

The movie rating files contain over 100 million ratings from 480 thousand randomly-chosen, anonymous Netflix customers over 17 thousand movie titles. The data were collected between October, 1998 and December, 2005 and reflect the distribution of all ratings received during this period. The ratings are on a scale from 1 to 5 (integral) stars. To protect customer privacy, each customer id has been replaced with a randomly-assigned id.

♦ Approach ♦

○ This large dataset involves just two columns one is user ID other is rating and the entries without rating are movie ID and another dataset involving movie id , movie name .

○ Data exploration - counted movies with different rating . Made differnt tuple using null rating values which basically depicted different users rating a single movie . Found the mean rating for sinle movie.

○ Data cleaning - created a benchmark and drop the user id who doesnt rate much also those movies who arent rated that much.

○ Made a sparse matrix of rows as user id and columns as movie id and entries as ratings

○ Used scikit surprise library for svd ,reader , dataset and to prevent from overfitting cross validate.we will be validating on 3 folds.USED SVD and found RMSE.

○ For a particular found the movies with 5 rating.Create a shallow copy for the movies dataset. remove movies rated less often. get the full data nd build a trainset using build_full_dataset. feed in svd.Estimate rating score a user can give to movie. Print top 10 movies with highest score.

♦ TECH ♦

SVD- Single value decomposition

Matrix Factorisation

Stochastic Gradient Descent
