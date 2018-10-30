import os

indir = '/home/aditya/Downloads/ShortAnswerGrading_v2.0/data/scores'
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        if f == 'ave':
            log = open(os.path.join(root, f), 'r')
            root2 = "/home/aditya/PycharmProjects/short-answer-grader/Train Data/"
            root3 = "/home/aditya/PycharmProjects/short-answer-grader/Test Data/"
            newFile = open(root2 + root.rsplit('/', 1)[-1] + " score", "w+")
            newFile2 = open(root3 + root.rsplit('/', 1)[-1] + " score", "w+")
            lines = log.readlines()
            newFile.writelines([item for item in lines[:-6]])
            newFile2.writelines([item for item in lines[-6:]])

indir = '/home/aditya/Downloads/ShortAnswerGrading_v2.0/data/raw'
for root, dirs, filenames in os.walk(indir):
    for f in filenames:
        log = open(os.path.join(root, f), 'r')
        root2 = "/home/aditya/PycharmProjects/short-answer-grader/Train Data/"
        root3 = "/home/aditya/PycharmProjects/short-answer-grader/Test Data/"
        newFile = open(root2 + f, "w+")
        newFile2 = open(root3 + f, "w+")
        lines = log.readlines()
        newFile.writelines([item for item in lines[:-6]])
        newFile2.writelines([item for item in lines[-6:]])