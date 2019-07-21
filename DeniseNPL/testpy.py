from bs4 import BeautifulSoup
import requests

body = {'query':'rune'}
r = requests.post('http://dictionary.com/browse/pizza',data=body)
soup = BeautifulSoup(r.text, 'html.parser')
#print(soup.find('head').find('meta').find('meta'))
print(soup.find_all('meta')[1])
#print(soup.head.meta)
#print(soup.prettify())
