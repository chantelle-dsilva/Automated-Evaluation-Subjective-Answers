from featureExtraction import *
import ridgeModel
from preProcess import *


def read_questions(tt_dir):

    questions_file = tt_dir + '/questions'

    questions = {}
    f = open(questions_file, 'rb')
    for line in f:
        line = line.strip()
        question_num = line.split(' ')[0]
        question_text = ' '.join(line.split(' ')[1:])
        questions[question_num] = question_text
    f.close()

    return questions


def read_reference_answers(tt_dir):

    ref_answers_file = tt_dir + '/reference answers'

    ref_answers = {}
    f = open(ref_answers_file, 'rb')
    for line in f:
        line = line.strip()
        answer_num = line.split(' ')[0]
        answer_text = ' '.join(line.split(' ')[1:])
        ref_answers[answer_num] = answer_text
    f.close()

    return ref_answers


def read_student_responses(tt_dir, question_num, corpus):

    student_responses_file = \
        tt_dir + '/' + str(question_num)

    student_responses = []
    f = open(student_responses_file, 'rb')
    for line in f:
        if line.strip():
            line = line.strip()
            response_num = line.split(' ')[0]
            response_text = ' '.join(line.split(' ')[1:])
            process1 = remove_all_non_printable(response_text)
            process2 = remove_all_non_characters(process1)
            process3 = spellcheck(process2)
            process4 = process3.lower()
            process5 = remove_multispaces(process4)
            #process6 = remove_stopwords(process5)
            #print "P6: "+process6
            student_responses.append(process5)
            corpus.append(process5)
    f.close()

    return student_responses, corpus


def read_scores(tt_dir, question_num):

    scores_file = tt_dir + '/' + str(question_num) + ' score'

    scores = []
    f = open(scores_file, 'rb')
    for line in f:
        line = line.strip()
        score = float(line)
        scores.append(score)
    f.close()

    return scores


def read_train_data():

    train_data = {}    
    corpus = []
    questions = read_questions('Train Data')
    ref_answers = read_reference_answers('Train Data')
    
    for question_num in questions:

        student_responses, corpus = read_student_responses('Train Data', question_num, corpus)
        scores = read_scores('Train Data', question_num)
        train_data[question_num] = (questions[question_num],
                                    ref_answers[question_num],
                                    student_responses,
                                    scores)

    return train_data
    

def extract_features(question, ref_answer, student_responses, student_response):

    #print student_response
    sim_alignment, cov_alignment, parse_results = \
                            sts_alignment(ref_answer, student_response)
    
    q_demoted_sim_alignment, q_demoted_cov_alignment, _ = \
                            sts_alignment(ref_answer, student_response,
                                          parse_results,
                                          question)

    sim_cvm = sts_cvm(ref_answer, student_response, parse_results)
    
    q_demoted_sim_cvm = sts_cvm(ref_answer, student_response,
                                parse_results,
                                question)
    
    lr = length_ratio(ref_answer, student_response, parse_results)

    d2v = testd2v(ref_answer, student_response)

    tfidf_score = tfidf(ref_answer, student_responses)

    textinfo = calculate_text_information(student_response)

    norm1, norm2 = question_description_similarities(question, student_response)

    feature_vector = (sim_alignment, cov_alignment,
                      q_demoted_sim_alignment, q_demoted_cov_alignment,
                      sim_cvm,
                      q_demoted_sim_cvm,
                      lr, d2v, tfidf_score,
                      textinfo, norm1, norm2)
    
    return feature_vector
    

def construct_train_examples(train_data):

    train_examples = []
    
    for question_num in train_data:
        data_for_this_question = train_data[question_num]
        question = data_for_this_question[0]
        ref_answer = data_for_this_question[1]
        student_answers = data_for_this_question[2]
        scores = data_for_this_question[3]
        for i in xrange(len(student_answers)):
            print "Question num: " + question_num
            features = extract_features(question, ref_answer, student_answers,
                                        student_answers[i])
            score = scores[i]
            train_examples.append((features, score))

    return train_examples
    
    
def train_grader(train_examples):
    
    model = ridgeModel.train_model([item[0] for item in train_examples],
                                   [item[1] for item in train_examples])
                                   
    return model


def read_test_data():

    questions = read_questions('Test Data')
    ref_answers = read_reference_answers('Test Data')
    corpus = []

    for question_num in questions:
        f = open('Test Data/' + question_num + ' predicted score', 'w')
        print 'calculating scores for question number: ' + question_num + '\n'
        student_responses, corpus = read_student_responses('Test Data', question_num, corpus)
        for student_response in student_responses:
            score = grade(questions[question_num], ref_answers[question_num], student_responses, student_response, grader)
            print score
            f.write('%.1f\n' % score)
        f.close()


def grade(question, ref_answer, student_responses, student_response, grader):

    features = extract_features(question, ref_answer, student_responses, student_response)
    score = ridgeModel.predict(grader, [features])[0]

    return score


print 'reading train data from files...'
train_data = read_train_data()
print 'done.'
print

print 'extracting features and constructing training examples...'
train_examples = construct_train_examples(train_data)
print 'done.'
print

print 'training the grading model...'
grader = train_grader(train_examples)
print 'done.'
print

print 'testing the grading model...'
read_test_data()
print 'done.'