import sys

### Load spaCy's English NLP model


def howTo():
  import spacy
  import webbrowser
  from googlesearch import search

  print("Please wait a moment")
  nlp = spacy.load('en_core_web_lg')
## The text we want to examine
  text = input("What do you want to learn how to do? ")


### Parse the text with spaCy
### Our 'document' variable now contains a parsed version of text.
  document = nlp(text)

### print out all the named entities that were detected
  found = False
  line = ""

  for token in document:
    print(token.pos_)
    if(found == False):
      if(token.pos_ == "VERB"):
        found = True
    else:
      line = line + " " + token.text

  types = input("look for video version? [y/n] ")
  if(types == "y"):
    line = "how to " + line + " video"
  else:
    line = "how to " + line + " instructions"

  for j in search(line, tld='co.in', num=1, stop=1,pause=2):
    go = input("open in new window " + line + "? [y/n] ")
    if(go == "y"):
      print("opening " + j)
      webbrowser.open_new(j)
      break

def searchAll():
  import spacy
  import webbrowser
  from googlesearch import search

  print("Please wait a moment")
  nlp = spacy.load('en_core_web_lg')
  nlp = spacy.load('en_core_web_lg')
  text = input("What do you want to google? ")
  document = nlp(text)
  line = " "

  for token in document:
    line = line + " " + token.text
  for j in search(line,tld='co.in',num=1,stop=1,pause=2):
    go = input("open in new window " + line + "? [y/n] ")
    if(go == "y"):
      print("opening " + j)
      webbrowser.open_new(j)
      break

def dictionary():
  from bs4 import BeautifulSoup
  import requests
  import re

  word = input("What would you like to define? ")
  body = {'query':'rune'}
  r = requests.post('http://dictionary.com/browse/' + word,data=body)
  soup = BeautifulSoup(r.text, 'html.parser')
  defi = soup.find_all('meta')[1]
  print(" ")
  #print(re.sub(r'.*content=', "", str(defi)))
  print(str(defi)[15:-31])

def translate():
  from bs4 import BeautifulSoup
  import requests
  import re

  word = input("What would you like to translate to English? ")
  body = {'query':'rune'}
  list_words = word.split()
  link =""
  if(len(list_words) > 0):
    x = 0
    sz = len(list_words)
    while(x < sz - 1):
      link += list_words[x] +"%20"  #maybe detect lang then use deepl
      x+=1
    link += list_words[x]
  print(link)
  #r = requests.post('http://www.deepl.com/en/' + link,data=body)
  #soup = BeautifulSoup(r.text, 'html.parser')

kg = True

#while(kg):
#  choice = input("""
#  Would you like to:
#  search google [1]
#  Learn how to...[2]
#  define a word [3]
#  """)

#  if(choice == "1"):
#    searchAll()
#  elif(choice == "2"):
#    howTo()
#  elif(choice == "3"):
#    dictionary()
#  else:
#    print("Not a valid input")

#  print(" ")
#  conti = input("Keep going [y/n] ? " )
#  if(conti == "n"):
#    kg = False

#make jokes
#math
#weather
#call alexa
translate()
