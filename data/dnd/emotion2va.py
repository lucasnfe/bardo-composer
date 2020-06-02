import os

MAPPING = {"happy": "1,1", "calm": "1,0", "agitated": "0,1", "suspense": "0,0"}

for file in os.listdir("."):
    name, ext = os.path.splitext(file)
    if ext == ".txt":
        sentiment_lines = ""
        with open(file) as f_read:
            for line in f_read:
                example = line.rsplit(":", 1)
                x,y = example[0], example[1].replace('\n', '')
                y = MAPPING[y]
                sentiment_lines += x + ":" + y + "\n"

        with open(file, "w") as f_write:
            f_write.write(sentiment_lines)
