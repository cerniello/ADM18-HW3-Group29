# Search_Engine_Airbnb

## Create different kind of Search Engine in order to retrieve documents from airbnb in the Texas state. 
3rd homework for Alogithm of Data Minig @LaSapienza University

<p align="center">
  <img width="460" height="300" src="https://www.kajola.net/wp-content/uploads/2017/11/AirBnB.jpg">
</p>

### Data format: csv 

## Structure : 
 - average_rate_per_night
 - bedrooms_count 
 - city
 - date_of_listing 
 - description  
 - latitude
 - longitude
 - title
 - url

### Uploaded data from csv and cleaned them

In order to analize good data, duplicates been deleted and also nan values related to longitude and latitude, that will be used subsequently.

### Create Documents - Preprocessing

For each row in the csv, a new document is been created, in tsv format (each document corresponds to an ad in airbnb) .
The access to data will be only through these documents.

## Scripts from functions.py
  
  0. remove_step : this first function helps us to further clean data, in particular for having all text parts of the document (text + description) with the actual words, with no punctuation; so, for example, all stopword are removed and all words became tokens using PorterStemmer methods from nltk library.  
  
  1. create_vocabulary_and_ii1 : code to build the vocabulary and the first inverted index. For this function it's been required to pass through each document, checking all words.
  The two dictionaries created have been saved in a pickle file, so the function doesn't return anything.
  
  2. search_engine_1 : basing on the query given in input, it's been calculated the result of all the documents where all its words are present. The method creates the dataframe and returns the first 5 documents of it.
  
  3. inverted_index_TFIDF : it exploits 2 methods, reduce_doc_list and compute_ii2_TFIDF.
  
  reduce_doc_list takes care of all the words (term_id), appending all documents containing the term_id into its list (the list is the value into the dict). This is the first step for computing the tfidf. 
  
  With function compute_ii2_TFIDF we create the TFIDF following the theoretical formulas learned in classes.
  
  4. search_engine_2 :
   The method, using a combination of ii1 and ii2, looks up only on documents that contains all the words of the query.
   After took every useful document with each tfidf, computing cosine distances with query array (absolute frequencies of the terms), it returns the list with each useful document and the related similarity. 
   
  The method is fast because, computing a smaller ii2 for each query, there is no need to create big structures.
 first_k_documents takes the list returned from the search engine and show the first k documents into a dataframe.
 
  5. search_engine_3 : the last search engine exploit the first inverted index, but computing a different score.
  It's been considerated, as new informations to get the most good document, the average per night, the number of bedrooms and the distance for each document retrieved by the first search engine.
  Each documents as a new score, but still between 0 and 1.
  Everything is saved into a heap and the top k documents are showed with the same Pandas dataframe standard, as before. 
  
## Bonus 

  6. houses_map : the method takes every possible location with a maximum distance and prints all the airbnb into the radius of the maximum distance.
  
  
 
  
 
 
  
  
