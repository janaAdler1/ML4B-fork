# Related Work

## Jan
https://hal.archives-ouvertes.fr/file/index/docid/725096/filename/dynam-proceedings.pdf#page=20

-> generelle Schwierigkeiten bei der Einordnung in Parteien

https://sentic.net/predicting-political-sentiments-of-voters-from-twitter.pdf

-> Ähnlich zu unserem Thema:
Analyse der Wähler (nicht Parteien) in Indien anhand derer Aktivitäten auf Twitter (durch Neuronale Netze und durch Multinomiale logistische Regression {-> Statistische Methode,  kein ML}

https://arxiv.org/pdf/2202.00396.pdf
-> Sentiment Analyse 

https://ojs.aaai.org/index.php/ICWSM/article/view/14139/13988
-> User Classification auf Basis von ML

https://medium.com/analytics-vidhya/predicting-political-orientation-with-machine-learning-be65c950d366
-> Vorhersage der politischen Orientierung mit ML

https://reader.elsevier.com/reader/sd/pii/S1110016821000806?token=0ECA4C78E9DD0ECBE1635C362A776B6ABF830420581757A5832B21829B5BDFFB704DF7D5883082A110DA3FE6024CB857&originRegion=eu-west-1&originCreation=20220518040840
-> Text Klassifizierung mit ML, allerdings auf Englisch. Dennoch gute Einblicke 

https://link.springer.com/article/10.1007/s00500-020-05209-8
-> Text Klassifizierung mit ML

## Jana
- https://jisajournal.springeropen.com/articles/10.1186/s13174-018-0089-0
- https://towardsdatascience.com/building-a-user-classifier-using-twitter-data-283dfd0c0e59 --> nur basic idee
- https://pdf.co/blog/classifying-tweets-via-machine-learning-in-python
- https://www.tensorflow.org/tutorials/keras/text_classification -> Text Klassifikation inkl. Code
- https://towardsdatascience.com/understanding-nlp-word-embeddings-text-vectorization-1a23744f7223#:~:text=Word%20Embeddings%20or%20Word%20vectorization,into%20numbers%20are%20called%20Vectorization. -> Vektorisierung von Text
- https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7 
- https://github.com/CharaZhu/Twitter-Sentiment-Analysis
- https://paperswithcode.com/task/text-classification
- https://towardsdatascience.com/binary-classification-of-imdb-movie-reviews-648342bc70dd

## Sergei
- https://jdvala.github.io/blog.io/thesis/2018/05/11/German-Preprocessing.html
- https://towardsdatascience.com/nlp-classification-in-python-pycaret-approach-vs-the-traditional-approach-602d38d29f06
## Problems and surprises
 - Dataset with 8 GB
 - Classification into parties can become relatively difficult if they have overlapping views (e.g. SPD and CDU)
 - Data set must first be prepared at great effort
 - Tweets are German and we need to adapt German letters (see: ppc.py)
 - More examples with sentiment analysis 
## What is important for project
- NLP text Vectorization
- Text classification through supervised machine learning
