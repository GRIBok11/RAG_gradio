import mimetypes

file_path = "moo2_manual.pdf"
mime_type, encoding = mimetypes.guess_type(file_path)

if mime_type is not None:
    print(f"Тип файла: {mime_type}")
else:
    print("Не удалось определить тип файла")
