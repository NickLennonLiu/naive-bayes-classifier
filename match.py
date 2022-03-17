import re

def count_url(raw):
    url_pattern = r'(https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    return len(re.findall(url_pattern, raw))

def count_email(raw):
    email_pattern = r'([a-z0-9_\.\+-]+)@([\da-z\.-]+)\.([a-z\.]{2,6})'
    return len(re.findall(email_pattern, raw))

def count_phone(raw):
    patt = r'(?:(?:\(?(?:00|\+)([1-4]\d\d|[1-9]\d?)\)?)?[\-\.\ \\\/]?)?((?:\(?\d{1,}\)?[\-\.\ \\\/]?){0,})(?:[\-\.\ \\\/]?(?:#|ext\.?|extension|x)[\-\.\ \\\/]?(\d+))?'
    return len(re.findall(patt, raw))

def process(raw):
    url, email, phone = count_url(raw), count_email(raw), count_phone(raw)
    print("url: {}, email: {}, phone: {}".format(url, email, phone))