# Define the business problem.
There is need to validate movie reviews to find out are the written comments inline with ratings from 1 to 4 (bad) and 7 to 10 (good) given by the movie watchers.

If there is strong connection between current textual feedback and reviews there could be possibility to collect feedback text only and find reviews based on text.

# Decide approach what you will use to solve described problem and create your hypothesis.
There are ca 25000 feedbacks available. To find out are there any common words and phrases in feedbacks
WordCloud can be used to list and visualise common words and phrases. To validate feedback text can be
processed with NLTK and converted into categorical features with Count Vectoriser. The most performant classifier in this case was GaussianNB.

# Decide what results do you need.

1. Reviews rate histogram to understand reviews distribution in existing feedback.
2. WordCloud and most frequent words list sorted by frequency.

# Evaluate results and decide does selected approach solve your business problem. If you thing that your
results show that your business problem is not resolved, provide explanation why you thing it can not be
solved with selected approach.
Seems that here is enough textual feedback to train classifier. Although the distribution of especially extremly bad and extremly good movies is unbalanced with the rest with cases predictions to be most likely a bit better than the actual value.
It is practically possible to predict  correct results even though the correctness is way higher if you just focus on wheather the film was good or not and not go into detail.