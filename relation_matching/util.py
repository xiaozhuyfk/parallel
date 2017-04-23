import codecs
import json
import subprocess


# Read a file
# filename is the path of the file, string type
# returns the content as a string
def readFile(filename, mode="rt"):
    # rt stands for "read text"
    fin = contents = None
    try:
        fin = open(filename, mode)
        contents = fin.read()
    finally:
        if (fin != None): fin.close()
    return contents


# Write 'contents' to the file
# 'filename' is the path of the file, string type
# 'contents' is of string type
# returns True if the content has been written successfully
def writeFile(filename, contents, mode="wt"):
    # wt stands for "write text"
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True


def codecsReadFile(filename, mode="rt", encoding='utf-8'):
    # rt stands for "read text"
    f = contents = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        contents = f.read()
    finally:
        if (f != None): f.close()
    return contents


def codecsWriteFile(filename, contents, mode="wt", encoding='utf-8'):
    f = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        f.write(contents)
    finally:
        if (f != None): f.close()
    return True


def loadJson(filename, mode="rt", encoding='utf-8'):
    f = None
    d = None
    try:
        with open(filename, mode) as f:
            d = json.load(f)
    finally:
        if (f != None): f.close()
    return d


def dumpJson(filename, contents, mode="wt", encoding='utf-8'):
    f = None
    try:
        with open(filename, mode) as f:
            json.dump(contents, f, indent=4)
    finally:
        if (f != None): f.close()
    return True


def codecsLoadJson(filename, mode="rt", encoding='utf-8'):
    f = None
    d = None
    try:
        with codecs.open(filename, mode, encoding) as f:
            d = json.load(f)
    finally:
        if (f != None): f.close()
    return d


def codecsDumpJson(filename, contents, mode="wt", encoding='utf-8'):
    f = None
    try:
        with codecs.open(filename, mode, encoding) as f:
            json.dump(contents, f, indent=4)
    finally:
        if (f != None): f.close()
    return True


"""return a tuple with recall, precision, and f1 for one example"""


def computeF1(goldList, predictedList):
    """Assume all questions have at least one answer"""
    if len(goldList) == 0:
        if len(predictedList) == 0:
            return (1, 1, 1)
        else:
            return (0, 0, 0)
    """If we return an empty list recall is zero and precision is one"""
    if len(predictedList) == 0:
        return (0, 1, 0)
    """It is guaranteed now that both lists are not empty"""

    precision = 0
    for entity in predictedList:
        if entity in goldList:
            precision += 1
    precision = float(precision) / len(predictedList)

    recall = 0
    for entity in goldList:
        if entity in predictedList:
            recall += 1
    recall = float(recall) / len(goldList)

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * recall * precision / (precision + recall)
    return (recall, precision, f1)


def kstem(stem):
    cmd = ['java',
           '-classpath',
           'kstem.jar',
           'org.lemurproject.kstem.KrovetzStemmer',
           '-w',
           stem]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = p.communicate()
    result = out.split(" ")[1][:-1]
    return result

def brief_result(path, new_path):
    import ast

    prefix = path[:-4]
    correct_path = prefix + "_correct.txt"
    partial_path = prefix + "_partial.txt"
    wrong_path = prefix + "_wrong.txt"
    entity_path = prefix + "_noentity.txt"

    lines = codecsReadFile(path).strip().split("\n")
    codecsWriteFile(new_path, "")
    codecsWriteFile(correct_path, "")
    codecsWriteFile(partial_path, "")
    codecsWriteFile(wrong_path, "")
    codecsWriteFile(entity_path, "")

    for line in lines:
        sections = line.strip().split('\t')
        content = []
        query = sections[0]
        gold = [n.strip() for n in ast.literal_eval(sections[1])]
        answer = [n.strip() for n in ast.literal_eval(sections[2])]

        if len(answer) > 10:
            answer_new = list(set(gold) & set(answer))
            if len(answer_new) == 0:
                answer = answer[:10] + ['...']
            else:
                answer = answer_new
        content += [query, str(gold), str(answer)]

        if len(sections) > 3:
            f1 = float(sections[3])
            gold_rel = sections[4]
            answer_rel = sections[5]
            gold_subject = sections[6]
            answer_subject = sections[7]
            content += [str(f1), gold_rel, answer_rel, gold_subject, answer_subject]
        message = "\t".join(content) + "\n"
        codecsWriteFile(new_path, message, 'a')

        if len(sections) > 3:
            f1 = float(sections[3])
            if f1 == 1.0:
                codecsWriteFile(correct_path, message, 'a')
            elif f1 > 0.0:
                codecsWriteFile(partial_path, message, 'a')
            else:
                codecsWriteFile(wrong_path, message, 'a')
        else:
            codecsWriteFile(entity_path, message, 'a')



if __name__ == '__main__':
    # print edit_distance('this is a house', 'this is not a house')
    # sftp_get("/home/hongyul/Python-2.7.11.tgz", "/Users/Hongyu1/Desktop/Python.tgz")
    # sftp_get_r("/home/hongyul/query", "/Users/Hongyu1/Desktop")
    # sftp_put("/Users/Hongyu1/Desktop/Python.tgz", "/home/hongyul/haha.tgz")
    # print sftp_execute("../init_env/bin/python indri.py name_of_collection_activity")
    # print sftp_listdir("/home/hongyul/")
    # get_filenames()
    # sftp_put("/data/dump.tar.gz", "/home/hongyul/aqqu/testresult/dump")
    # test()
    import globals
    globals.read_configuration('config.cfg')
    config_options = globals.config
    base = config_options.get('DEFAULT', 'base')
    result = base + '/test_result/result_pairwise+joint+pretrain.txt'
    new = base + '/test_result/result_pairwise+joint+pretrain_brief.txt'

    brief_result(result, new)
