import os
import email
import string

def get_body(payload):
    body = ""
    if payload.is_multipart():
        for subpayload in payload.get_payload(decode=True):
            body = body + get_body(subpayload)
    else:
        body = body + payload.get_payload(decode=True)
    return body

def parse_raw_spam():
    for root, dirs, files in os.walk('../Resources/enron6/spam'):
        with open("../Resources/spam.txt", "w") as output_file:
            for file in files:
                with open('../Resources/enron6/spam/' + file) as email_file:
                    email_object = email.message_from_file(email_file)
                    output = {}
                    output["subject"] = filter(lambda x: x in set(string.printable), str(email_object["Subject"]).replace('\r\n',' '))
                    output["body"] = filter(lambda x: x in set(string.printable), get_body(email_object).replace('\r\n',' '))
                    output_file.write(output.__str__())
                    output_file.write("\n")
def parse_raw_ham():
    for root, dirs, files in os.walk('../Resources/enron6/ham'):
        with open("../Resources/ham.txt", "w") as output_file:
            for file in files:
                with open('../Resources/enron6/ham/' + file) as email_file:
                    email_object = email.message_from_file(email_file)
                    output = {}
                    output["subject"] = filter(lambda x: x in set(string.printable), str(email_object["Subject"]).replace('\r\n',' '))
                    output["body"] = filter(lambda x: x in set(string.printable), get_body(email_object).replace('\r\n',' '))
                    output_file.write(output.__str__())
                    output_file.write("\n")
