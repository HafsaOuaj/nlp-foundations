from sklearn.feature_extraction.text import CountVectorizer # impliment the bag of words model tokenizer + counts word frequency


vectorizer = CountVectorizer()

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]
X = vectorizer.fit_transform(corpus)

print(f"The words found by the analyzer during the fit:\n {vectorizer.get_feature_names_out()}")
print(f"get the int corresponding to the 'document' term: {vectorizer.vocabulary_.get('document')}")
print(f"any term that is out of vocab will be completely ignored in the fit:\n {vectorizer.transform(['Something completely new.']).toarray()}")
analyze = vectorizer.build_analyzer()

analyze("This is a text document to analyze.") == (
    ['this', 'is', 'text', 'document', 'to', 'analyze'])


print("\n--- Bi-grams ---\n To preserve the order of words, we can use n-grams:")

# char_wb - create n-grams only from text inside word boundaries
# char - create n-grams from all characters including spaces and punctuation
bigram_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                    token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
print(analyze('Bi-grams are cool!')) 
analyze('Bi-grams are cool!') == (
    ['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])


ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])

ngram_vectorizer.get_feature_names_out()

ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(['jumpy fox'])
ngram_vectorizer.get_feature_names_out()