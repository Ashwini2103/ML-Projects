#ML-Projects

Extraction of Keywords from PDF Document

Steps Performed in extraction of Keywords:-
1. Extraction of PDF File from Link http://bit.ly/epo_keyword_extraction_document
2. Using Text Normalization of Natural Language Processing (NLP)
	where punctuation, stopwords, converting to lowercase is done.
3. POS(Parts-of-Speech) Tagging is also done where words are assigned to Noun,Pronoun,Adjective categories respectively.
4. Lemmatization of words is performed . Lemmatization means considering ( Ex: Application, Applications ) as a single word Application 
	itself throughout the document.
5. Apply POS tags to Lemmatized words.
6. Filtering out words which belong to POS =['NN','NNS','NNP','NNPS','JJ','JJR','JJS','VBG','FW']
   NN  - Noun, singular or mass
   NNS - Noun, plural
   NNP - Proper noun, singular
   NNPS- Proper noun, plural
   JJ  - Adjective
   JJR - Adjective, comparative
   JJS - Adjective, superlative
   VBG - Verb, gerund or present participle
   FW  - Foreign word
7. Assigned weights to words using CountVectorizer and TF-IDF technique. 