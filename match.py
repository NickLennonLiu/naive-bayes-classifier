import re
import numpy as np

def count_url(raw):
    url_pattern = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    return len(re.findall(url_pattern, raw))

def count_email(raw):
    email_pattern = r'([a-z0-9_\.\+-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})'
    return len(re.findall(email_pattern, raw))

def count_phone(raw):
    patt = r'(\d*-)*\d+'
    return len(re.findall(patt, raw))

def process(raw):
    url, email, phone = count_url(raw), count_email(raw), count_phone(raw)
    # print("url: {}, email: {}, phone: {}".format(url, email, phone))
    return [url, email, phone]

def content_feature(raw):
    # raw: list of texts
    return np.array([process(i) for i in raw])


if __name__ == "__main__":
    test_raw = ["www.a.com", "1880-0131-229", "a@b.c", "nonsense"]
    print(content_feature(test_raw))
    print(count_phone("nonsense"))
