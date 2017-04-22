import logging
import xml.etree.ElementTree as etree
import nltk.data
import os
import json
import string
import urllib2
import requests
import re
from nltk.tokenize import sent_tokenize

from util import (
    codecsWriteFile,
    codecsReadFile,
    dumpJson,
    loadJson
)

logger = logging.getLogger(__name__)

class WikiExtractor(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.data = None
        self.abstract = None
        self.data_xml = config_options.get('Wiki', 'data')
        self.abstract_xml = config_options.get('Wiki', 'abstract')
        self.wiki_dir = config_options.get('Wiki', 'wiki-dir')
        self.enwiki_dir = config_options.get('Wiki', 'enwiki-dir')

        self.file_pattern = self.enwiki_dir + "enwiki-latest-pages-articles%d.xml"
        self.support_dir = config_options.get('Wiki', 'support-sentence-dir')
        self.title_dir = config_options.get('Wiki', 'title-sentence-dir')

    @staticmethod
    def init_from_config(config_options):
        return WikiExtractor(config_options)

    def wiki_data_on_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return os.path.isfile(path)

    def load_wiki_data_from_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return codecsReadFile(path)

    def store_wiki_data(self, dataset, query, subject, text):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        path = query_dir + subject
        codecsWriteFile(path, text)

    def get_support_sentence_with_title(self, sid, oid):
        import modules
        if not oid.startswith("m."): return []

        subject_name = modules.wiki_url[sid]
        object_name = modules.wiki_url[oid]
        if subject_name is None or object_name is None:
            return []

        result = []
        result_path = self.title_dir + subject_name
        result_path = result_path.encode('utf-8')
        if (os.path.isfile(result_path)):
            sentences = codecsReadFile(result_path).strip().split("\n")
            for sent in sentences:
                if ("[[" not in sent and "]]" not in sent):
                    continue
                for occur in sent.split("[[")[1:]:
                    idx = occur.find("]]")
                    entity = occur[:idx]
                    if ("File:" in entity or
                        "Image:" in entity or
                        "Category:" in entity or
                        "Wikipedia:" in entity or
                        "Template:" in entity):
                        continue
                    if ('|' in entity):
                        entity = entity[:entity.find('|')]
                    if ('#' in entity):
                        entity = entity[:entity.find('#')]
                    if ('/' in entity):
                        entity = entity.replace('/', '|')
                    entity = entity.replace(" ", "_")
                    if entity == object_name:
                        result.append(sent.encode("utf-8"))
                        break
        return result


    def get_support_sentence_with_pair(self, sid, oid):
        import modules
        if not oid.startswith("m."):
            return []

        subject_name = modules.wiki_url[sid]
        object_name = modules.wiki_url[oid]
        if subject_name is None or object_name is None:
            return []

        result = []
        for i in xrange(27):
            idx = i+1
            result_path = self.support_dir + str(idx) + '/' + subject_name
            result_path = result_path.encode("utf-8")
            if (os.path.isfile(result_path)):
                sentences = codecsReadFile(result_path).strip().split('\n')
                for sent in sentences:
                    if ("[[" not in sent and "]]" not in sent):
                        continue
                    for occur in sent.split("[[")[1:]:
                        idx = occur.find("]]")
                        entity = occur[:idx]
                        if ("File:" in entity or
                            "Image:" in entity or
                            "Category:" in entity or
                            "Wikipedia:" in entity or
                            "Template:" in entity):
                            continue
                        if ('|' in entity):
                            entity = entity[:entity.find('|')]
                        if ('#' in entity):
                            entity = entity[:entity.find('#')]
                        if ('/' in entity):
                            entity = entity.replace('/', '|')
                        entity = entity.replace(" ", "_")
                        if entity == object_name:
                            result.append(sent.encode("utf-8"))
                            break
        return result

    def extract_support_sentence(self, idx):
        logger.info("Start extracting support sentences for partition %d", idx)
        prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
        xml_path = self.file_pattern % idx
        result_path = self.support_dir + str(idx) + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        for event, elem in etree.iterparse(
                xml_path,
                events=('start', 'end', 'start-ns', 'end-ns')
            ):
            if (event == 'end' and elem.tag[len(prefix):] == 'page'):

                try:
                    title = elem.find(prefix + "title").text.replace(' ', '_')
                    title = title.replace('/', '|').encode('utf-8')
                    logger.info("Processing wiki page: %s", title)
                    title_path = self.title_dir + title
                    codecsWriteFile(title_path, "")

                    revision = elem.find(prefix + "revision")
                    text = revision.find(prefix + "text").text

                    content = ""
                    for part in re.split(ur'</ref>|/>', text, flags=re.UNICODE):
                        idx = part.find('<ref')
                        if idx >= 0:
                            content += part[:idx]
                        else:
                            content += part
                    text = content

                    content = ""
                    for part in re.split(ur'}}', text, flags=re.UNICODE):
                        idx = part.find('{{')
                        if idx >= 0:
                            content += part[:idx]
                        else:
                            content += part
                    text = content

                    content = ""
                    for part in re.split(ur'-->', text, flags=re.UNICODE):
                        idx = part.find('<!--')
                        if idx >= 0:
                            content += part[:idx]
                        else:
                            content += part
                    text = content

                    paragraphs = text.strip().split("\n")
                    sentences = [sent_tokenize(p) for p in paragraphs if p]
                    sentences = [s for p in sentences for s in p]

                    for sent in sentences:
                        if ("[[" not in sent and "]]" not in sent):
                            continue
                        entities = []
                        for occur in sent.split("[[")[1:]:
                            idx = occur.find("]]")
                            entity = occur[:idx]
                            if ("File:" in entity or
                                "Image:" in entity or
                                "Category:" in entity or
                                "Wikipedia:" in entity or
                                "Template:" in entity):
                                continue
                            if ('|' in entity):
                                entity = entity[:entity.find('|')]
                            if ('#' in entity):
                                entity = entity[:entity.find('#')]
                            if ('/' in entity):
                                entity = entity.replace('/', '|')
                            entities.append(entity)

                        for entity in entities:
                            if not entity:
                                continue
                            filename = entity.replace(" ", "_")
                            filepath = result_path + filename.encode('utf-8')
                            if os.path.isfile(filepath):
                                codecsWriteFile(filepath, sent + "\n", 'a')
                            else:
                                codecsWriteFile(filepath, sent + "\n")
                        codecsWriteFile(title_path, sent + "\n", 'a')
                except:
                    logger.info("Error processing")
        logger.info("Finished extracting support sentences for partition %d", idx)

    def extract_wiki_page(self, dataset, query, subject):
        logger.info("Extracting wiki from question %d: %s" % (query.id, query.utterance))
        if self.wiki_data_on_disk(dataset, query, subject):
            return self.load_wiki_data_from_disk(dataset, query, subject)
        else:
            if self.abstract is None:
                self.abstract = set([])
                for event, elem in etree.iterparse(
                        self.abstract_xml,
                        events=('start', 'end', 'start-ns', 'end-ns')
                ):
                    if (event == 'end') and (elem.tag == 'title'):
                        self.abstract.add(elem.text[11:].lower())
                logger.info("Load abstract complete.")
            d = {}
            prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
            if subject in self.abstract:
                for event, elem in etree.iterparse(
                    self.data_xml,
                    events=('start', 'end', 'start-ns', 'end-ns')
                ):
                    if (event == 'end' and elem.tag[len(prefix):] == 'page'):
                        title = elem.find(prefix + "title")
                        revision = elem.find(prefix + "revision")
                        text = revision.find(prefix + "text")

                        if title.text.lower() != subject:
                            continue

                        self.store_wiki_data(dataset, query, subject, text)
                        return text

                        """
                        paragraphs = text.text.strip().split("\n")
                        sentences = [tokenizer.tokenize(p) for p in paragraphs if p]
                        sentences = [s for p in sentences for s in p]

                        for object in candidate.objects:
                            d[object] = []
                            for sent in sentences:
                                if subject in sent and object in sent:
                                    d[object].append(sent)
                        """


class WikiAPIExtractor(object):

    def __init__(self, config_options):
        self.config_options = config_options
        self.url = "https://en.wikipedia.org/w/api.php"
        self.wiki = "https://en.wikipedia.org/wiki/index.html"
        self.wiki_dir = config_options.get('Wiki', 'wiki-dir')
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def init_from_config(config_options):
        return WikiAPIExtractor(config_options)

    def store_wiki_data(self, dataset, query, subject, text):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        if not os.path.exists(query_dir):
            os.makedirs(query_dir)
        path = query_dir + subject
        dumpJson(path, text)

    def wiki_data_on_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return os.path.isfile(path)

    def load_wiki_data_from_disk(self, dataset, query, subject):
        query_dir = self.wiki_dir + dataset + '/' + str(query.id) + '/'
        path = query_dir + subject
        return loadJson(path)

    def extract_wiki_page(self, dataset, query, subject, sid):
        #logger.info("Extracting wiki from question %d: %s" % (query.id, query.utterance))
        if self.wiki_data_on_disk(dataset, query, sid):
            return self.load_wiki_data_from_disk(dataset, query, sid)
        parameter = {
            "action": "query",
            "format": "json",
            "titles": subject,
            "prop": "revisions",
            "rvprop": "content"
        }
        r = requests.get(self.url, params = parameter)

        key, value = r.json()["query"]["pages"].popitem()
        if key == "-1":
            return []

        text = value["revisions"][0]["*"].lower()
        paragraphs = text.strip().split("\n")
        sentences = [self.tokenizer.tokenize(p) for p in paragraphs if p]
        sentences = [s for p in sentences for s in p]
        self.store_wiki_data(dataset, query, sid, sentences)

        return sentences

    def extract_test(self, subject):
        parameter = {
            "action": "query",
            "format": "json",
            "titles": subject,
            "prop": "revisions",
            "rvprop": "content"
        }
        r = requests.get(self.url, params = parameter)

        key, value = r.json()["query"]["pages"].popitem()
        if key == "-1":
            return []

        text = value["revisions"][0]["*"].lower()
        paragraphs = text.strip().split("\n")
        sentences = [self.tokenizer.tokenize(p) for p in paragraphs if p]
        sentences = [s for p in sentences for s in p]
        return sentences


if __name__ == '__main__':
    abstract_xml = "/home/hongyul/AMA/wiki/enwiki/enwiki-latest-pages-articles1.xml"
    prefix = "{http://www.mediawiki.org/xml/export-0.10/}"
    count = 0
    for event, elem in etree.iterparse(abstract_xml, events=('start', 'end', 'start-ns', 'end-ns')):
        if (event == 'end' and elem.tag[len(prefix):] == 'page'):
            title = elem.find(prefix + "title")
            revision = elem.find(prefix + "revision")
            text = revision.find(prefix + "text").text

            content = ""
            for part in re.split(ur'</ref>|/>', text, flags=re.UNICODE):
                idx = part.find('<ref')
                if idx >= 0:
                    content += part[:idx]
                else:
                    content += part
            text = content

            content = ""
            for part in re.split(ur'}}', text, flags=re.UNICODE):
                idx = part.find('{{')
                if idx >= 0:
                    content += part[:idx]
                else:
                    content += part
            text = content

            content = ""
            for part in re.split(ur'-->', text, flags=re.UNICODE):
                idx = part.find('<!--')
                if idx >= 0:
                    content += part[:idx]
                else:
                    content += part
            text = content

            paragraphs = content.strip().split("\n")
            sentences = [sent_tokenize(p) for p in paragraphs if p]
            sentences = [s for p in sentences for s in p]
            for sent in sentences:
                if ("[[" not in sent and "]]" not in sent):
                    continue
                entities = []
                for occur in sent.split("[[")[1:]:
                    idx = occur.find("]]")
                    entity = occur[:idx]
                    if ("File:" in entity or "Image:" in entity or "Category:" in entity):
                        continue
                    if ('|' in entity):
                        entity = entity[:entity.find('|')]
                    entities.append(entity)

                for entity in entities:
                    filename = entity.replace(" ", "_")
                    print filename,
                print
                print sent

            """
            ref_start = ur'<ref'
            ref_end = ur'</ref>|/>'
            iter_start = list(re.finditer(re.compile(ref_start), text))
            iter_end = list(re.finditer(re.compile(ref_end), text))

            replace = []
            assert(len(iter_start) == len(iter_end))

            for i in xrange(len(iter_start)):
                start = iter_start[i].start(0)
                end = iter_end[i].end(0)
                print text[start:end]
                print
                replace.append(text[start:end])
            """
            #for rep in replace:
            #    text = text.replace(rep, "")
            #print text

            if title.text == "Anarchism":
                break
    #wiki = WikiAPIExtractor(None)
    #print wiki.extract_test("Barack Obama")

