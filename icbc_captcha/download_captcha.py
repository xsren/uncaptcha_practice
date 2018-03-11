import os
import requests


def get_captca(img_name):
    captcha_url = 'https://epass.icbc.com.cn/servlet/ICBCVerificationCodeImageCreate?randomId=1509526710324308014&height=36&width=90&appendRandom=1509526741726'

    headers1 = {

        'Accept': '*/*',
        'Referer': 'https://epass.icbc.com.cn/servlet/com.icbc.inbs.person.servlet.Verifyimage3?randomKey=1509526710324308014&imageAlt=%E7%82%B9%E5%87%BB%E5%9B%BE%E7%89%87%E5%8F%AF%E5%88%B7%E6%96%B0&imgheight=36&safePassId=safeEdit1&safePassName=logonCardPass&imgwidth=90&appendRandom=1509526741726&flushflag=1',
        'Accept-Language': 'zh-CN',
        'User-Agent': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729)',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'epass.icbc.com.cn',
        'Connection': 'Keep-Alive'
    }

    req = requests.session()
    img = req.get(captcha_url,headers=headers1, verify=False)
    with open('{}.png'.format(img_name), 'wb') as f:
        f.write(img.content)

for i in range(10):
    get_captca(i)