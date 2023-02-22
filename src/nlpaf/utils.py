# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:12:51 2020

"""
import importlib
import itertools
import os
import hashlib
import pathlib
import pkgutil

import nltk
import re
from nltk.stem.porter import PorterStemmer


def hashfile(path, blocksize=65536):
    afile = open(path, "rb")
    hasher = hashlib.md5()
    buf = afile.read(blocksize)
    while len(buf) > 0:
        hasher.update(buf)
        buf = afile.read(blocksize)
    afile.close()
    return hasher.hexdigest()


def findDuplicates(parentFolder):
    # Dups in format {hash:[names]}
    dups = {}
    for dirName, subdirs, fileList in os.walk(parentFolder):
        print("Scanning %s..." % dirName)
        for filename in fileList:
            path = os.path.join(dirName, filename)
            file_hash = hashfile(path)
            if file_hash in dups:
                dups[file_hash].append(path)
            else:
                dups[file_hash] = [path]

    result = {}
    for k, v in dups.items():
        if len(v) > 1:
            result[k] = v

    return result


def save_json(df, name):
    with open("data/" + name + ".json", "w") as w:
        w.write(df.to_json(orient="records"))


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        top_ids = topic.argsort()[: -n_top_words - 1 : -1]
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i] for i in top_ids])
        print(message)
    print()


def tokenize(text):
    tokens = [
        w.lower()
        for w in nltk.word_tokenize(text)
        if not bool(re.search("[*\W*|_]", w))
    ]
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems


def flatten_list(lst):
    return list(itertools.chain(*lst))


def load_module(module_name: str):
    return importlib.import_module(module_name)


def get_cls_from_path(path: str):
    module_name, class_name = path.rsplit(".", 1)

    cls = getattr(importlib.import_module(module_name), class_name)
    return cls, class_name


def load_sub_packages(package):
    prefix = package.__name__ + "."
    # print(f"prefix {prefix}")
    for importer, modname, ispkg in pkgutil.iter_modules(
        package.__path__, prefix
    ):
        # print(f"Found submodule {modname} (is a package: {ispkg})" )
        module = load_module(modname)
        # print (f"Imported {module}")


def create_folder(folder_path):
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)


def file_exists(file_path):
    return os.path.isfile(file_path)
