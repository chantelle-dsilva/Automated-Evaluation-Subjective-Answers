from __future__ import division

from align import *
from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import pandas as pd
import gensim
import numpy as np
import random
import zlib
import string

embeddings = {}

def load_embeddings(file_name):

    embeddings = {}

    input_file = open(file_name, 'r')
    for line in input_file:
        tokens = line.split('\t')
        tokens[-1] = tokens[-1].strip()
        for i in xrange(1, len(tokens)):
            tokens[i] = float(tokens[i])
        embeddings[tokens[0]] = tokens[1:-1]

    return embeddings


def vector_sum(vectors):

    n = len(vectors)
    d = len(vectors[0])

    s = []
    for i in xrange(d):
        s.append(0)
    s = np.array(s)

    for vector in vectors:
        s = s + np.array(vector)

    return list(s)


def cosine_similarity(vector1, vector2):
    return 1 - spatial.distance.cosine(vector1, vector2)


def sts_alignment(sentence1, sentence2,
                  parse_results=None,
                  sentence_for_demoting=None):

    if sentence1 and sentence2:

        if parse_results == None:
            sentence1_parse_result = parseText(sentence1)
            sentence2_parse_result = parseText(sentence2)
            parse_results = []
            parse_results.append(sentence1_parse_result)
            parse_results.append(sentence2_parse_result)
        else:
            sentence1_parse_result = parse_results[0]
            sentence2_parse_result = parse_results[1]


        sentence1_lemmatized = lemmatize(sentence1_parse_result)
        sentence2_lemmatized = lemmatize(sentence2_parse_result)

        lemmas_to_be_demoted = []
        if sentence_for_demoting != None:
            if len(parse_results) == 2:
                sentence_for_demoting_parse_result = \
                                    parseText(sentence_for_demoting)
                parse_results.append(sentence_for_demoting_parse_result)
            else:
                sentence_for_demoting_parse_result = parse_results[2]


            sentence_for_demoting_lemmatized = \
                                lemmatize(sentence_for_demoting_parse_result)

            sentence_for_demoting_lemmas = \
                            [item[3] for item in sentence_for_demoting_lemmatized]

            lemmas_to_be_demoted = \
                    [item.lower() for item in sentence_for_demoting_lemmas \
                                if item.lower() not in stop_words+punctuations]

        alignments = align(sentence1, sentence2,
                           sentence1_parse_result, sentence2_parse_result)[0]

        sentence1_lemmas = [item[3] for item in sentence1_lemmatized]
        sentence2_lemmas = [item[3] for item in sentence2_lemmatized]

        sentence1_content_lemmas = \
                [item for item in sentence1_lemmas \
                          if item.lower() not in \
                                stop_words+punctuations+lemmas_to_be_demoted]

        sentence2_content_lemmas = \
                [item for item in sentence2_lemmas \
                        if item.lower() not in \
                                 stop_words+punctuations+lemmas_to_be_demoted]

        if sentence1_content_lemmas == [] or sentence2_content_lemmas == []:
            return (0, 0, parse_results)

        sentence1_aligned_content_word_indexes = \
            [item[0] for item in alignments if \
                    sentence1_lemmas[item[0]-1].lower() not in \
                                    stop_words+punctuations+lemmas_to_be_demoted]

        sentence2_aligned_content_word_indexes = \
            [item[1] for item in alignments if \
                    sentence2_lemmas[item[1]-1].lower() not in \
                                    stop_words+punctuations+lemmas_to_be_demoted]

        sim_score = (len(sentence1_aligned_content_word_indexes) + \
                     len(sentence2_aligned_content_word_indexes)) / \
                                            (len(sentence1_content_lemmas) + \
                                              len(sentence2_content_lemmas))

        coverage = len(sentence1_aligned_content_word_indexes) / \
                                               len(sentence1_content_lemmas)

        return (sim_score, coverage, parse_results)


def sts_cvm(sentence1, sentence2,
            parse_results,
            sentence_for_demoting=None,):

    global embeddings

    if embeddings == {}:
        print 'loading embeddings...'
        embeddings = \
           load_embeddings('Resources/EN-wform.w.5.cbow.neg10.400.subsmpl.txt')
        print 'done'

    sentence1_parse_result = parse_results[0]
    sentence2_parse_result = parse_results[1]

    sentence1_lemmatized = lemmatize(sentence1_parse_result)
    sentence2_lemmatized = lemmatize(sentence2_parse_result)

    lemmas_to_be_demoted = []
    if sentence_for_demoting != None:
        sentence_for_demoting_parse_result = parse_results[2]

        sentence_for_demoting_lemmatized = \
                            lemmatize(sentence_for_demoting_parse_result)

        sentence_for_demoting_lemmas = \
                        [item[3] for item in sentence_for_demoting_lemmatized]

        lemmas_to_be_demoted = \
    			[item.lower() for item in sentence_for_demoting_lemmas \
        					if item.lower() not in stop_words+punctuations]

    sentence1_lemmas = [item[3].lower() for item in sentence1_lemmatized]
    sentence2_lemmas = [item[3].lower() for item in sentence2_lemmatized]

    #sentence1_lemmas[:] = sorted(sentence1_lemmas)
    #sentence2_lemmas[:] = sorted(sentence2_lemmas)

    if sentence1_lemmas == sentence2_lemmas:
        return 1

    sentence1_content_lemma_embeddings = []
    for lemma in sentence1_lemmas:
        if lemma.lower() in stop_words+punctuations+lemmas_to_be_demoted:
            continue
        if lemma.lower() in embeddings:
            sentence1_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])


    sentence2_content_lemma_embeddings = []
    for lemma in sentence2_lemmas:
        if lemma.lower() in stop_words+punctuations+lemmas_to_be_demoted:
            continue
        if lemma.lower() in embeddings:
            sentence2_content_lemma_embeddings.append(
                                            embeddings[lemma.lower()])

    if sentence1_content_lemma_embeddings == \
                       sentence2_content_lemma_embeddings:
        return 1
    elif sentence1_content_lemma_embeddings == [] or \
         sentence2_content_lemma_embeddings == []:
        return 0

    sentence1_embedding = vector_sum(sentence1_content_lemma_embeddings)
    sentence2_embedding = vector_sum(sentence2_content_lemma_embeddings)

    return cosine_similarity(sentence1_embedding, sentence2_embedding)


def length_ratio(sentence1, sentence2, parse_results):

    sentence1_parse_result = parse_results[0]
    sentence2_parse_result = parse_results[1]

    sentence1_lemmatized = lemmatize(sentence1_parse_result)
    sentence2_lemmatized = lemmatize(sentence2_parse_result)

    sentence1_lemmas = [item[3] for item in sentence1_lemmatized]
    sentence2_lemmas = [item[3] for item in sentence2_lemmatized]

    sentence1_content_lemmas = \
            [item for item in sentence1_lemmas \
                      if item.lower() not in \
                            stop_words+punctuations]

    sentence2_content_lemmas = \
            [item for item in sentence2_lemmas \
					if item.lower() not in \
                             stop_words+punctuations]

    if sentence2_content_lemmas == []:
        return len(sentence1_lemmas) / len(sentence2_lemmas)

    return len(sentence1_content_lemmas) / len(sentence2_content_lemmas)


def testd2v (ref_answer, student_response):

    d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')

    ref = ref_answer.split()
    stud = student_response.split()

    #inference hyper-parameters
    start_alpha=0.01
    infer_epoch=1000

    ref_v = d2v_model.infer_vector(ref, alpha=start_alpha, steps=infer_epoch)
    stud_v = d2v_model.infer_vector(stud, alpha=start_alpha, steps=infer_epoch)
    return (cosine_similarity(ref_v,stud_v))


def tfidf (ref_answer,student_responses):

    student_responses.append(ref_answer)
    vectoriser = TfidfVectorizer(ngram_range=(2,3), sublinear_tf=True, use_idf =True, \
                                 stop_words = 'english')

    train_dm = vectoriser.fit_transform(student_responses)

    svd = TruncatedSVD()
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    lsa = lsa.fit_transform(train_dm)
    pd.DataFrame(lsa, index = student_responses, columns = ["component_1","component_2"]).tail(5)
    similarity = np.asarray(np.asmatrix(lsa) * np.asmatrix(lsa).T)
    df = pd.DataFrame(similarity,index=student_responses, columns=student_responses).tail(5)

    similarity =  df.iloc[4,-2]

    del student_responses[-1]  #should i leave in the ref_answer to make it more important in the corpus or something (for idf)?
    return similarity


def calculate_text_information(student_response):
    random_text = word_generator(len(student_response))
    if len(student_response) > 0:
        info_value = len(zlib.compress(student_response, 9)) / len(zlib.compress(random_text, 9))
    else:
        info_value = 0
    return info_value


def word_generator(size=6, chars=string.ascii_uppercase):
    return ''.join(random.choice(chars) for x in range(size))


def question_description_similarities(question, student_response):
    question_infovalue = len(zlib.compress(question, 9))
    random_text = word_generator(len(student_response))
    if len(student_response) > 0:
        answer_infovalue = len(zlib.compress(question + student_response, 9))
        answer_infovalue_dummy = len(zlib.compress(question + random_text, 9))

        answer_infovalue_length_raw = answer_infovalue - question_infovalue
        if len(student_response) != 0:
            answer_infovalue_length_norm = (answer_infovalue - question_infovalue) / len(student_response)
        else:
            answer_infovalue_length_norm = 0

        if (question_infovalue - answer_infovalue_dummy) != 0:
            answer_infovalue_length_norm2 = (question_infovalue - answer_infovalue) / (
            question_infovalue - answer_infovalue_dummy)
        else:
            answer_infovalue_length_norm2 = 0
    else:
        # answer_infovalue_length_raw = -1
        answer_infovalue_length_norm = -1
        answer_infovalue_length_norm2 = -1

    return answer_infovalue_length_norm, answer_infovalue_length_norm2
