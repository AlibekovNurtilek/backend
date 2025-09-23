from pytube import YouTube

# Ссылка на видео
url = "https://youtu.be/ZuhUONEJLRc?si=Lm64pwz97guAlrWV"

# Создаем объект YouTube
yt = YouTube(url)

# Получаем название
print(yt.title)
