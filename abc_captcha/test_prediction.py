import requests
import base64
import json
import io
from PIL import Image
from predict_img import Predict

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def down_abc_captcha():
    url = "https://perbank.abchina.com/EbankSite/LogonImageCodeAct.do?r=0.8297626506212761"
    response = requests.request(method="GET", url=url, verify=False)
    encoded = base64.b64encode(response.content)
    resp_cookies = response.cookies
    cookie = {}
    for co in resp_cookies:
        cookie[co.name] = co.value
    return json.dumps(cookie, ensure_ascii=False), str(encoded, 'utf-8')


def check_abc_captcha(cookies, code):
    url = "https://perbank.abchina.com/EbankSite/VerifyPicCodeAct.do"

    querystring = {"picCode": code, "r": "0.8649984243166056"}

    headers = {
        'host': "perbank.abchina.com",
        'connection': "keep-alive",
        'accept': "application/json, text/javascript, */*; q=0.01",
        'x-requested-with': "XMLHttpRequest",
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36",
        'dnt': "1",
        'referer': "https://perbank.abchina.com/EbankSite/startup.do",
        'accept-encoding': "gzip, deflate, br",
        'accept-language': "zh-CN,zh;q=0.8,en;q=0.6",
        'cache-control': "no-cache"
    }
    response = requests.request("GET", url, headers=headers, params=querystring,cookies=json.loads(cookies),verify=False)
    result = response.text
    return result


if __name__ == '__main__':
    foo = Predict()
    flag = 0
    start = 0
    for i in range(2000):
        cookies, img_base64 = down_abc_captcha()
        img_bytes = base64.b64decode(img_base64)
        img_file = io.BytesIO(img_bytes)
        code = foo.predict_img(img_file)
        resp = check_abc_captcha(cookies, code)
        if json.loads(resp).get('errorCode',None) == "0000":
            with open('temp/{}_{}.jpg'.format(start, code), 'wb') as f:
                f.write(img_bytes)
            start += 1
            flag += 1
        else:
            pass
    print('acc is', flag/2000)
