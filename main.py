import torch
import sounddevice as sd
import time  ## <- Bu qatorni o'chirib tashlang ! Dastur ishlash vaqtini bilish uchun yozilgan.

language = "uz"
model_id = 'v3_uz'
sample_rate = 48000
speaker = 'dilnavoz'
put_accent = True
put_yoo = True

# GPU ni tekshirish
# Agar GPU mavjud bo'lsa, uni ishlatadi, aks holda CPU ni ishlatadi
# GPU borligi serverga bog'liq, parametrlarida yozilgan bo'ladi.Server xususiyatlarini ko'rib chiqing.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Modelni sozlash
model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language=language, speaker=model_id)
model.to(device)

# TTS funksiyasi, matnni ovozga aylantiradi.
# Bu qism funksiya ichida bo'lishi kerak.Sababi ,shunda model har safar load qilinmaydi va tezroq ishlaydi.
# Agar bu qismni funksiya ichiga olinmasa, har safar modelni load qilish va ovozni generatsiya qilish uchun vaqt sarflanadi.
def tts_func(text):
    start_time = time.time()  ## <- Bu qatorni o'chirib tashlang ! Dastur ishlash vaqtini bilish uchun yozilgan.
    
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate, put_accent=put_accent)
    
    # Bu 3 ta qatorni ham o'chiring ! Dastur ishlash vaqtini bilish uchun yozilgan.
    end_time = time.time() ## O'chiring
    duration = end_time - start_time  ## O'chiring
    print(f"Vaqt: {duration:.2f} sekund ({duration * 1000:.2f} millisekund)") ## O'chiring
    
    print(text)
    sd.play(audio, sample_rate)
    sd.wait()  # Oldin ishlatilgan time.sleep ijroni bloklaydi, wait esa ijro tugashini kutib turadi va vaqtdan yutadi.


# test
texts = "Ipoteka krediti yoki ikkilamchi bozordan 3 ,4 va 5 xonali uy-joy sotib olish uchun ipoteka krediti olishingiz mumkin.Ipoteka krediti olish uchun sizga quyidagilar kerak bo'ladi:"

tts_func(texts)


# O'chiring deyilgan joylarini o'chirsangiz yana biroz vaqtdan yutasiz,hozir ular uchun ham vaqt sarflab turibdi.
# print(torch.cuda.is_available())  - > GPU bor yoki yo'q ekanligini tekshirish uchun(True yoki False qaytaradi).Agar aslida bor bo'lsa-yu False qaytarsa,serverni GPU sifatida ishlatishga ruxsat berilmagan.
