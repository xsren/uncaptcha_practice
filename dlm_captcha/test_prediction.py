import io
from predict_img import Predict
from fetch_captcha import down_captcha, veri_captcha


bar = Predict()
flag = 0
for i in range(200):
    try:
        cookie, img = down_captcha()
        # b64_img = base64.b64encode(img)
        # to_img = base64.b64decode(b64_img)
        file = io.BytesIO(img)
        # img_serial = cookie['PHPSESSID']
        img_name = bar.predict_img(file)
        status = veri_captcha(cookie, img_name)
        if status == 'ok':
            print('success')
            flag += 1
        else:
            print('failed')
    except:
        pass
print(flag)
