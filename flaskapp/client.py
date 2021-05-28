import requests
r = requests.get('https://lab5-test-xyz.herokuapp.com/net')
print(r.status_code)
print(r.text)
r = requests.get('https://lab5-test-xyz.herokuapp.com/data_to')
print(r.status_code)
print(r.text)

# импортируем нужные модули
import os
from io import BytesIO
import base64


img_data = None
# создаем путь к файлу (для кроссплатформенности, например)
path = os.path.join('./static', 'image0008.png')
# читаем файл и енкодируем его в строку base64
with open(path, 'rb') as fh:
    img_data = fh.read()
    b64 = base64.b64encode(img_data)
# создаем json словарь, который
# отправляется на сервер в виде json строки
# преобразование делает сама функция отправки запроса post
jsondata = {'imagebin':b64.decode('utf-8')}
res = requests.post('https://lab5-test-xyz.herokuapp.com/apinet', json=jsondata)
if res.ok:
    print(res.json())


try:
    r = requests.get('http://127.0.0.1:5000/apixml')
    print(r.status_code)
    if(r.status_code!=200):
        exit(1)
    print(r.text)
except:
    exit(1)

# запрос к синусу
r = requests.get('http://127.0.0.1:5000/sin')
print(r.status_code)
print(r.text)

# try:
#     r = requests.get('http://localhost:5000/imgfilter')
#     print(r.status_code)
#     if(r.status_code!=200):
#         exit(1)
#     print(r.text)
# except:
#     exit(1)