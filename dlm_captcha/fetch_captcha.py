import requests


def down_captcha():
    url1 = 'http://www.dailianmeng.com/xinyong/q/111.html'
    headers = {
        'Accept': 'text/html, application/xhtml+xml, image/jxr, */*',
        'Accept-Language': 'zh-CN',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.dailianmeng.com',
        'Connection': 'Keep-Alive'
    }
    req = requests.session()
    req1 = req.get(url=url1, headers=headers)
    cookies = req1.cookies
    url2 = 'http://www.dailianmeng.com/xinyong/captcha.html'
    req2 = req.get(url=url2)
    return {c.name: c.value for c in cookies}, req2.content


def veri_captcha(cookies, code):
    val_headers = {
        'Accept': 'text/html, application/xhtml+xml, image/jxr, */*',
        'Referer': 'http://www.dailianmeng.com/xinyong/q/111.html',
        'Accept-Language': 'zh-CN',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko',
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept-Encoding': 'gzip, deflate',
        'Host': 'www.dailianmeng.com',
        'Connection': 'Keep-Alive',
        'Pragma': 'no-cache'
    }
    val_url = 'http://www.dailianmeng.com/xinyong/q/111.html'
    payload = 'SearchForm%5BverifyCode%5D={}&yt0='.format(int(code))
    veri = requests.post(url=val_url, headers=val_headers, data=payload, cookies=cookies)
    if '身份证:111的贷款信用查询结果' in veri.text:
        return 'ok'
    else:
        return 'no'


if __name__ == '__main__':
    cookie, img = down_captcha()
    print(cookie['PHPSESSID'])